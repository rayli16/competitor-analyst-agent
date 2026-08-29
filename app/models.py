from typing import Literal

from pydantic import BaseModel, Field, create_model

class AnalysisRequest(BaseModel):
    """ 
    Input for analysis. Agent gathers its own data
    Users just input the name and optionally the URL and focus
    """
    competitor_name: str = Field(
        description="Name of the competitor to analyze"
        )
    competitor_url: str | None = Field(
        default=None, 
        description="URL of Competitor's official website, used to disambiguate the target (Optional)"
        )
    focus: str | None = Field(
        default=None,
        description="Optional area to focus on, e.g. 'pricing strategy', 'product'"
        )

class SourcedClaim(BaseModel):
    claim: str = Field(description="A specific factual claim about the competitor")
    source_url: str = Field(
        description="The exact URL from the provided sources that supports this claim, copied verbatim",
        json_schema_extra={"format": "uri"},
    )


class PricingClaim(BaseModel):
    # product's description allows a company-wide value too — otherwise a
    # signal like "quote-based, no published pricing" has nowhere honest to
    # go once product is required.
    product: str = Field(
        description=(
            "The specific product or product line this price applies to, or the "
            "company/brand name if the signal is company-wide rather than "
            "product-specific (e.g. 'quote-based, no published pricing')."
        )
    )
    price_signal: str = Field(description="The observed price or pricing indicator for this product")
    source_url: str = Field(
        description="The exact URL from the provided sources that supports this claim, copied verbatim",
        json_schema_extra={"format": "uri"},
    )


class AnalysisResponse(BaseModel):

    """ Structured output of the analysis """

    competitor_name: str = Field(
        description="The competitor analyzed, echoed back for traceability"
    )
    # Deliberately unsourced: a synthesis of many claims below, not one
    # verifiable fact — a single citation here would overclaim.
    summary: str = Field(
        description="High-level overview of the competitor's market position"
    )
    products: list[SourcedClaim] = Field(
        description="Product lines or key products offered, each with a source"
    )
    pricing_signals: list[PricingClaim] = Field(
        description="Per-product pricing signals, each with a source"
    )
    regional_presence: list[SourcedClaim] = Field(
        description="Markets, regions, or customer segments where they operate, each with a source"
    )
    # Deliberately unsourced: the model's own judgment/advice, not a
    # verifiable fact — there's nothing to cite a URL for.
    recommended_actions: list[str] = Field(
        description="Concrete suggested responses based on the findings"
    )


def build_sourced_response_type(urls: list[str]) -> type[AnalysisResponse]:
    # Literal[tuple(urls)] becomes a real JSON Schema "enum" — the model's
    # source_url is constrained to exactly these values, not any string.
    url_type = Literal[tuple(urls)]
    dynamic_claim = create_model(
        "SourcedClaim",
        __base__=SourcedClaim,
        source_url=(url_type, SourcedClaim.model_fields["source_url"]),
    )
    dynamic_pricing = create_model(
        "PricingClaim",
        __base__=PricingClaim,
        source_url=(url_type, PricingClaim.model_fields["source_url"]),
    )
    return create_model(
        "AnalysisResponse",
        __base__=AnalysisResponse,
        products=(list[dynamic_claim], AnalysisResponse.model_fields["products"]),
        pricing_signals=(list[dynamic_pricing], AnalysisResponse.model_fields["pricing_signals"]),
        regional_presence=(list[dynamic_claim], AnalysisResponse.model_fields["regional_presence"]),
    )