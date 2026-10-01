"""Strict wire schemas. Missing values are explicit; unknown keys are rejected."""
from __future__ import annotations
import json
import re
from datetime import date
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

Number = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Ratio = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]

class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

class InvoiceItem(StrictModel):
    name: str = Field(min_length=1)
    amount_without_tax: Number | None
    tax_rate: Ratio | None
    tax_amount: Number | None

class InvoiceExtractionResult(StrictModel):
    invoice_date: str | None
    taxpayer_id: str | None
    buyer_name: str | None
    items: list[InvoiceItem]
    missing_fields: list[str]

    @field_validator("invoice_date")
    @classmethod
    def valid_date(cls, value):
        if value is not None and (not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) or date.fromisoformat(value).isoformat() != value):
            raise ValueError("invoice_date must be an actual YYYY-MM-DD date")
        return value

    @field_validator("taxpayer_id")
    @classmethod
    def valid_id(cls, value):
        if value is not None and not re.fullmatch(r"[A-Z0-9]{18}", value):
            raise ValueError("taxpayer_id must contain 18 uppercase letters or digits")
        return value

    @field_validator("buyer_name")
    @classmethod
    def valid_buyer(cls, value):
        if value is not None and not value.strip():
            raise ValueError("empty buyer_name must be null")
        return value

    @model_validator(mode="after")
    def consistent_missing_fields(self):
        expected = {key for key in ("invoice_date", "taxpayer_id", "buyer_name") if getattr(self, key) is None}
        if not self.items:
            expected.add("items")
        for index, item in enumerate(self.items):
            for key in ("amount_without_tax", "tax_rate"):
                if getattr(item, key) is None:
                    expected.add(f"items.{index}.{key}")
        if set(self.missing_fields) != expected or len(set(self.missing_fields)) != len(self.missing_fields):
            raise ValueError(f"missing_fields must equal {sorted(expected)}; tax_amount is optional")
        return self

class TaxAmountParameters(StrictModel):
    amount_without_tax: Number
    tax_rate: Ratio

class DeductionParameters(StrictModel):
    tax_amount: Number
    deductible_ratio: Ratio

class TaxCall(StrictModel):
    tool_name: Literal["calculate_tax_amount"]
    parameters: TaxAmountParameters

class DeductionCall(StrictModel):
    tool_name: Literal["calculate_vat_deduction"]
    parameters: DeductionParameters

ToolCallAction = Annotated[TaxCall | DeductionCall, Field(discriminator="tool_name")]

class ExtractionResponse(StrictModel):
    action: Literal["extract"]
    data: InvoiceExtractionResult
    reason: None

class ToolResponse(StrictModel):
    action: Literal["call_tool"]
    data: ToolCallAction
    reason: None

class FallbackResponse(StrictModel):
    action: Literal["fallback"]
    data: None
    reason: Literal["insufficient_information", "unsupported_request"]

Response = Annotated[ExtractionResponse | ToolResponse | FallbackResponse, Field(discriminator="action")]
RESPONSE_ADAPTER = TypeAdapter(Response)

def strict_json_loads(raw: str):
    def reject_constant(value):
        raise ValueError(f"Non-JSON number: {value}")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)

def validate_response(raw: str):
    return RESPONSE_ADAPTER.validate_python(strict_json_loads(raw))
