"""TOOL 1 – Calculator: arithmetic, percentages, GST, totals and averages.

Expressions are parsed with ``ast`` and evaluated by a whitelist interpreter:
no ``eval``/``exec``, no names, no calls, no attribute access.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from backend.exceptions import ToolInputError
from backend.tools.common import safe_tool_call

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_EXPR_LEN = 200


def safe_eval(expression: str) -> float:
    expr = (expression or "").strip()
    if not expr:
        raise ToolInputError("An expression is required.")
    if len(expr) > MAX_EXPR_LEN:
        raise ToolInputError("The expression is too long.")
    expr = (expr.replace(",", "").replace("×", "*").replace("÷", "/").replace("^", "**")
            .replace("₹", "").replace("$", "").replace("€", "").replace("£", ""))
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)  # 5% -> (5/100)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ToolInputError("The expression is not valid arithmetic.") from exc

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ToolInputError("That power is too large to compute safely.")
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ToolInputError("Division by zero.")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        raise ToolInputError("Only numbers and the operators + - * / // % ** ( ) are allowed.")

    try:
        result = ev(tree)
    except RecursionError as exc:
        raise ToolInputError("The expression is too deeply nested.") from exc
    if isinstance(result, complex) or not math.isfinite(result):
        raise ToolInputError("The result is not a finite real number.")
    return result


def _money(x) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _need(value, label: str):
    if value is None:
        raise ToolInputError(f"'{label}' is required for this operation.")
    return value


def run_calculator(operation: str, expression: str | None = None, numbers: list[float] | None = None,
                   amount: float | None = None, percent: float | None = None, other: float | None = None) -> dict:
    if operation == "evaluate":
        r = safe_eval(_need(expression, "expression"))
        return {"operation": operation, "expression": expression, "result": round(r, 10)}

    if operation == "percentage_of":      # percent% of amount
        a, p = _need(amount, "amount"), _need(percent, "percent")
        return {"operation": operation, "result": round(a * p / 100, 6), "explanation": f"{p}% of {a}"}

    if operation == "percentage_change":  # from amount (old) to other (new)
        old, new = _need(amount, "amount"), _need(other, "other")
        if old == 0:
            raise ToolInputError("Percentage change from zero is undefined.")
        return {"operation": operation, "result": round((new - old) / abs(old) * 100, 4),
                "explanation": f"change from {old} to {new} (percent)"}

    if operation == "percentage_share":   # amount is what % of other
        part, whole = _need(amount, "amount"), _need(other, "other")
        if whole == 0:
            raise ToolInputError("The total cannot be zero.")
        return {"operation": operation, "result": round(part / whole * 100, 4),
                "explanation": f"{part} is this percent of {whole}"}

    if operation in {"add_gst", "remove_gst"}:
        amt = _need(amount, "amount")
        rate = 18.0 if percent is None else percent
        if rate < 0 or rate > 100:
            raise ToolInputError("The GST rate must be between 0 and 100.")
        if operation == "add_gst":       # amount is GST-exclusive (net)
            gst = _money(amt * rate / 100)
            return {"operation": operation, "gst_rate_percent": rate, "net_amount": _money(amt), "gst_amount": gst,
                    "cgst": _money(gst / 2), "sgst": _money(gst / 2), "total_amount": _money(amt + gst)}
        base = _money(amt * 100 / (100 + rate))  # amount is GST-inclusive (gross)
        gst = _money(amt - base)
        return {"operation": operation, "gst_rate_percent": rate, "gross_amount": _money(amt), "net_amount": base,
                "gst_amount": gst, "cgst": _money(gst / 2), "sgst": _money(gst / 2)}

    if operation in {"total", "average"}:
        nums = numbers or []
        if not nums:
            raise ToolInputError("'numbers' must contain at least one value.")
        if len(nums) > 10_000:
            raise ToolInputError("Too many numbers.")
        total = math.fsum(nums)
        out = {"operation": operation, "count": len(nums), "total": round(total, 6), "minimum": min(nums), "maximum": max(nums)}
        if operation == "average":
            out["average"] = round(total / len(nums), 6)
            out["result"] = out["average"]
        else:
            out["result"] = out["total"]
        return out

    raise ToolInputError(f"Unknown operation '{operation}'.")


class CalculatorInput(BaseModel):
    operation: Literal["evaluate", "percentage_of", "percentage_change", "percentage_share",
                       "add_gst", "remove_gst", "total", "average"] = Field(
        description=(
            "evaluate: arithmetic 'expression'. percentage_of: 'percent'% of 'amount'. "
            "percentage_change: change from 'amount' (old) to 'other' (new). percentage_share: 'amount' as % of 'other'. "
            "add_gst: 'amount' is the price WITHOUT GST; returns GST and total. "
            "remove_gst: 'amount' is the price INCLUDING GST; returns the base price. "
            "total/average: list of 'numbers'."))
    expression: Optional[str] = Field(None, description="Arithmetic expression, e.g. '(1200*3)+450/2'. Only for 'evaluate'.")
    numbers: Optional[list[float]] = Field(None, description="Numbers for 'total' / 'average'.")
    amount: Optional[float] = Field(None, description="Main amount (see operation).")
    percent: Optional[float] = Field(None, description="Percentage or GST rate (default GST rate is 18).")
    other: Optional[float] = Field(None, description="Second value for percentage_change / percentage_share.")


@tool("calculator", args_schema=CalculatorInput)
def calculator(operation: str, expression: Optional[str] = None, numbers: Optional[list[float]] = None,
               amount: Optional[float] = None, percent: Optional[float] = None, other: Optional[float] = None) -> str:
    """Exact calculator for arithmetic, percentages, GST (add/remove, CGST/SGST split), totals and averages.
    ALWAYS use this tool for any numeric calculation instead of computing in your head."""
    return safe_tool_call("calculator", lambda: run_calculator(operation, expression, numbers, amount, percent, other))
