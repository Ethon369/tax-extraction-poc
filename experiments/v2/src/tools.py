"""Arithmetic demonstration tools; eligibility is supplied, never inferred."""
from __future__ import annotations
import re
from decimal import Decimal, ROUND_HALF_UP
from .schemas import ToolResponse, validate_response

SLOT_PATTERNS = {
    "amount_without_tax": r"(?:不含税(?:金额|价款|合计)?|未税(?:金额|价款)?|计税金额|净额)\s*[:：=为是]?\s*(\d+(?:\.\d+)?)\s*(%)?",
    "tax_rate": r"(?:税率|征收率)\s*[:：=为是]?\s*(\d+(?:\.\d+)?)\s*(%)?",
    "tax_amount": r"(?:税额|税金)\s*[:：=为是]?\s*(\d+(?:\.\d+)?)\s*(%)?",
    "deductible_ratio": r"(?:可抵扣比例|允许抵扣比例|抵扣比例|可抵扣份额)\s*[:：=为是]?\s*(\d+(?:\.\d+)?)\s*(%)?",
}

REGISTERED_TOOLS = {'calculate_tax_amount', 'calculate_vat_deduction'}

def request_boundary_issue(text: str, tool_name: str):
    requested = set(re.findall(r'(?<![A-Za-z0-9_])calculate_[A-Za-z0-9_]+(?![A-Za-z0-9_])', text))
    if requested - REGISTERED_TOOLS:
        return 'unregistered_tool_in_request'
    if requested and requested != {tool_name}:
        return 'requested_tool_mismatch'
    return None

def grounded_parameters(text: str, parameters: dict) -> bool:
    """Require a unique explicitly labelled source value for every tool parameter."""
    for key, value in parameters.items():
        values = set()
        for number, percent in re.findall(SLOT_PATTERNS[key], text):
            parsed = Decimal(number)
            if percent:
                parsed /= 100
            values.add(parsed)
        # Reject unlabeled alternatives in the same parameter clause too,
        # e.g. tax rate 6% OR 13%. Do not silently choose one candidate.
        for match in re.finditer(SLOT_PATTERNS[key], text):
            remainder = re.split(r'[，,。;；\n、]', text[match.end():], maxsplit=1)[0]
            if re.search(r'(?:或(?:者)?|至|到|~|～|—|-)\s*\d', remainder):
                return False
        if values != {Decimal(str(value))}:
            return False
    return True

def execute_checked(raw: str, source_text: str) -> dict:
    try:
        response = validate_response(raw)
    except (ValueError, TypeError) as exc:
        return {"status": "blocked", "reason": "invalid_output", "detail": str(exc)}
    if not isinstance(response, ToolResponse):
        return {"status": "not_requested"}
    issue = request_boundary_issue(source_text, response.data.tool_name)
    if issue:
        return {"status": "blocked", "reason": issue}
    parameters = response.data.parameters.model_dump()
    if not grounded_parameters(source_text, parameters):
        return {"status": "blocked", "reason": "ungrounded_or_ambiguous_parameters"}
    if response.data.tool_name == "calculate_tax_amount":
        result = Decimal(str(parameters["amount_without_tax"])) * Decimal(str(parameters["tax_rate"]))
    else:
        result = Decimal(str(parameters["tax_amount"])) * Decimal(str(parameters["deductible_ratio"]))
    return {"status": "executed", "tool_name": response.data.tool_name, "result_yuan": str(result.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))}
