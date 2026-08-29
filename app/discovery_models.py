from typing import Literal

from pydantic import BaseModel, Field, create_model

CandidateEntityType = Literal[
    "manufacturer",
    "trading_company",
    "distributor",
    "domestic_seller",
    "platform",
    "media_or_event",
    "uncertain",
]


class CandidateVerdict(BaseModel):
    company_name: str = Field(
        description=(
            "The company's name. If is_own_page is false, this is the company being "
            "DISCUSSED in third-party content, not the page's own operator."
        )
    )
    url: str = Field(
        description="The exact URL from the provided sources this candidate was found at, copied verbatim"
    )
    # Ordered before entity_type on purpose: this is the prerequisite judgment
    # — settle whose page it is before classifying what kind of business it is.
    is_own_page: bool = Field(
        description=(
            "True if this page IS the named company's own official site (their "
            "homepage, product page, or corporate site). False if this is "
            "THIRD-PARTY content that merely mentions or discusses a company — a "
            "news article, an investment firm's writeup, a directory/aggregator "
            "listing, a forum post. A page can describe a real manufacturer in "
            "detail and still be false here, if the page itself belongs to someone "
            "else (a publication, an investor, a directory)."
        )
    )
    entity_type: CandidateEntityType = Field(
        description=(
            "Classify by what the entity actually IS, not how it describes itself — "
            "distributors and trading companies often use words like 'supplier' in "
            "their own marketing copy even when they don't manufacture anything. "
            "If is_own_page is false, this must be 'media_or_event' or 'platform' "
            "(whichever matches the THIRD PARTY's own nature — a publication/investor "
            "vs. a directory/aggregator) — never 'manufacturer' or 'trading_company', "
            "regardless of what company the content discusses. If is_own_page is "
            "true, two signals matter most: (1) location — is this company based in "
            "China? A company based outside China selling this equipment is a "
            "distributor (a potential customer, not a competitor), regardless of "
            "self-description. (2) Does the source describe OWNING/PRODUCING a "
            "specific product line (factory details, model numbers, OEM/ODM "
            "capability) — that's 'manufacturer' — or SOURCING/REPRESENTING other "
            "companies' products? That's 'trading_company' if China-based and "
            "export-oriented, 'distributor' if based overseas. Use 'domestic_seller' "
            "for a China-based company that only appears to sell within China, with "
            "no export orientation evident. If the source doesn't give you enough to "
            "confidently pick one of the above — location unclear, or manufacturer "
            "vs. trading_company unclear — use 'uncertain' rather than guessing."
        )
    )
    evidence: str = Field(description="Brief note (1-2 sentences) on what in the source supports this classification")


def build_verdict_type(urls: list[str]) -> type[CandidateVerdict]:
    # Same Literal[tuple(urls)] -> enum trick as build_sourced_response_type
    # in app/models.py — url is structurally constrained to this batch's
    # real URLs, so a verdict can't cite a source that wasn't in front of it.
    url_type = Literal[tuple(urls)]
    return create_model(
        "CandidateVerdict",
        __base__=CandidateVerdict,
        url=(url_type, CandidateVerdict.model_fields["url"]),
    )


class ExtractionResult(BaseModel):
    verdicts: list[CandidateVerdict] = Field(
        description="Exactly one verdict per input result — every result must get an entry, none omitted"
    )


def build_extraction_result_type(urls: list[str]) -> type[ExtractionResult]:
    verdict_type = build_verdict_type(urls)
    return create_model(
        "ExtractionResult",
        __base__=ExtractionResult,
        verdicts=(list[verdict_type], ExtractionResult.model_fields["verdicts"]),
    )


class Company(BaseModel):
    canonical_name: str = Field(
        description=(
            "The company's most complete name — prefer a full legal/corporate name "
            "over a product-line brand name if both appear across the merged candidates"
        )
    )
    urls: list[str] = Field(
        description="All URLs belonging to this same real-world company, merged from possibly-different-looking candidates"
    )
    entity_type: CandidateEntityType = Field(
        description="The entity type for this company (the merged candidates should agree; if they conflict, prefer the more specific non-uncertain classification)"
    )
    merge_reasoning: str = Field(
        description=(
            "Why these URLs were judged to be the same company — e.g. shared address/phone, "
            "matching legal name, explicit cross-reference. If urls has only one entry, "
            "state 'single candidate, no merge'."
        )
    )


def build_company_type(urls: list[str]) -> type[Company]:
    url_type = Literal[tuple(urls)]
    return create_model(
        "Company",
        __base__=Company,
        urls=(list[url_type], Company.model_fields["urls"]),
    )


class DedupResult(BaseModel):
    companies: list[Company] = Field(description="The deduplicated, merged list of real companies")


def build_dedup_result_type(urls: list[str]) -> type[DedupResult]:
    company_type = build_company_type(urls)
    return create_model(
        "DedupResult",
        __base__=DedupResult,
        companies=(list[company_type], DedupResult.model_fields["companies"]),
    )


class DiscoveryRequest(BaseModel):
    our_products: str = Field(
        description=(
            "Free text describing our own product lines and go-to-market — the "
            "reference point scoring compares candidates against. Required: product "
            "overlap is meaningless without something to compare to."
        )
    )
    score_candidates: bool = Field(
        default=True,
        description="If true, score each discovered company against our_products. If false, return unscored discovery results only.",
    )


class ResolvedCompany(Company):
    # Never an LLM output_format — filled in by code after dedup, from
    # derive_domain_roots()/find_homepage_via_search() in discovery.py.
    primary_url: str = Field(description="The company's best-known homepage URL, resolved after deduplication")


Recommendation = Literal["deep_dive", "monitor", "exclude"]

# Literal instead of int with ge=1/le=10 — ge/le gets demoted to a
# description-only hint by the SDK's schema transform, while Literal
# becomes a real enum constraint, same as the URL fields below. Closes off
# out-of-range scores structurally instead of hoping the model behaves.
ScoreValue = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class CandidateScore(BaseModel):
    # Ties a score back to its company when several get scored in one
    # batched call — same "don't rely on list position" lesson as
    # check_full_coverage, just applied up front this time instead of
    # learning it the hard way again.
    company_url: str = Field(description="The primary_url of the company being scored, copied verbatim from the input")
    reasoning: str = Field(
        description="Brief justification for the three scores below, citing specific evidence from the profile search results"
    )
    product_overlap: ScoreValue = Field(
        description=(
            "How closely this company's products overlap with our_products. 10 = "
            "near-identical product lines and categories, 1 = no meaningful overlap. "
            "Judge by product category and function, not exact naming."
        )
    )
    export_orientation: ScoreValue = Field(
        description=(
            "How clearly this company is oriented toward exporting to large overseas "
            "distributors. 10 = strong, explicit export/overseas-market evidence, 1 = "
            "no evidence of export activity or appears domestic-only."
        )
    )
    info_availability: ScoreValue = Field(
        description=(
            "How much usable public information the profile searches turned up — a "
            "real homepage, specific product or company details, contact information. "
            "10 = rich, specific information; 1 = almost nothing findable beyond the name."
        )
    )
    llm_recommendation: Recommendation = Field(
        description="Your own recommendation given the scores and evidence above, independent of any fixed rule."
    )


def build_score_type(urls: list[str]) -> type[CandidateScore]:
    url_type = Literal[tuple(urls)]
    return create_model(
        "CandidateScore",
        __base__=CandidateScore,
        company_url=(url_type, CandidateScore.model_fields["company_url"]),
    )


class ScoringResult(BaseModel):
    scores: list[CandidateScore] = Field(
        description="Exactly one score per company provided — every company must get a score, none omitted"
    )


def build_scoring_result_type(urls: list[str]) -> type[ScoringResult]:
    score_type = build_score_type(urls)
    return create_model(
        "ScoringResult",
        __base__=ScoringResult,
        scores=(list[score_type], ScoringResult.model_fields["scores"]),
    )


class ScoredCompany(ResolvedCompany):
    # reasoning..llm_recommendation come straight from the LLM's CandidateScore.
    # rule_recommendation/rule_score/agrees are never LLM output — computed in
    # code by compute_recommendation() after the call returns.
    reasoning: str = Field(description="The model's justification for its scores")
    product_overlap: ScoreValue = Field(description="Product overlap with our_products, 1-10")
    export_orientation: ScoreValue = Field(description="Export/overseas-market orientation, 1-10")
    info_availability: ScoreValue = Field(description="Usable public information available, 1-10")
    llm_recommendation: Recommendation = Field(description="The model's own recommendation")
    rule_recommendation: Recommendation = Field(description="compute_recommendation()'s recommendation from the fixed thresholds")
    rule_score: float = Field(description="The weighted score behind rule_recommendation (0.0 if vetoed by info_availability <= 3)")
    agrees: bool = Field(description="Whether llm_recommendation and rule_recommendation match")


class DiscoveryResponse(BaseModel):
    # Union, not list[ResolvedCompany] — tried the parent type alone first
    # and it silently stripped every scoring field, since Pydantic
    # serializes against the declared field type, not the runtime type.
    # The union makes both real shapes explicit instead of quietly losing data.
    companies: list[ResolvedCompany] | list[ScoredCompany] = Field(
        description="Deduplicated candidate competitors found this run — scored if score_candidates was true"
    )
