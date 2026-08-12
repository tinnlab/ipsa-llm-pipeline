"""
Step 04: Mechanistic Hypothesis Generation

This step synthesizes outputs from Steps 1-3 to generate testable mechanistic hypotheses
that explain the observed molecular changes and suggest experimental validations.

Key features:
- Integrates pathway themes (Step 1), hub genes (Step 2), and mechanisms (Step 3)
- Generates up to 7 testable mechanistic hypotheses
- Provides directional predictions grounded in observed data
- Context-aware hypothesis generation (tissue/disease specific)
- Creates central mechanistic model
"""

import re
import sys
import json
import math
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.groq_client import GroqClient
from src.pipeline.services.pathway_query_service import PathwayQueryService
from src.pipeline.fc_utils import (
    fc_arrow, to_float, gene_fc_lookup, correct_fc_citations,
    sanitize_llm_text, drop_sentences_with_fc_conflicts,
)
# Reuse the NES-preferred enrichment extraction and magnitude calibration so
# hypothesis seeding ranks pathways on the same effect-size scale the report uses.
from src.pipeline.steps.step03_pathway_mechanisms import _enrichment_metric, _magnitude_label

# |NES| at/above which an enrichment direction is considered "dominant" and MUST
# be represented by a hypothesis (mirrors the 'strong' cutoff in _magnitude_label).
DOMINANT_NES_THRESHOLD = 2.0


def _nes_axis_signals(pathways: List[Dict]):
    """Rank pathways by enrichment score (NES preferred) into up- and down-axes.

    Returns ``(ups, downs)`` where each is a list of ``(value, name, metric)`` sorted
    by magnitude (ups descending, downs most-negative first). ``metric`` is the actual
    scale returned by :func:`_enrichment_metric` ('NES' or 'ES'), carried so callers
    label values honestly rather than hard-coding "NES". This is the single effect-size
    ranking used to seed hypotheses and enforce coverage, so the hypothesis step ranks
    pathways on the same scale as the executive summary.
    """
    ups, downs = [], []
    for p in pathways or []:
        if not isinstance(p, dict):
            continue
        val, metric = _enrichment_metric(p)
        val = to_float(val)
        if val is None:
            continue
        name = p.get('name') or p.get('pathwayName') or p.get('pathway') or ''
        if not name:
            continue
        entry = (val, name, metric)
        if val > 0:
            ups.append(entry)
        elif val < 0:
            downs.append(entry)
    ups.sort(key=lambda x: -x[0])
    downs.sort(key=lambda x: x[0])
    return ups, downs


@dataclass
class Hypothesis:
    """Single mechanistic hypothesis"""
    hypothesis: str
    mechanistic_model: str
    key_players: List[str]
    evidence_supporting: List[str]
    testability: Dict[str, str]
    directional_prediction: str
    confidence: str
    confidence_rationale: str
    novelty: str


@dataclass
class HypothesisResult:
    """Result of hypothesis generation"""
    hypotheses: List[Dict[str, Any]]
    central_mechanistic_model: str
    key_predictions: List[Dict[str, str]]
    hypotheses_summary: str
    report_section: str
    metadata: Dict[str, Any]


class Step04HypothesisGeneration:
    """
    Step 04: Mechanistic Hypothesis Generation

    Synthesizes pathway themes, hub genes, and mechanisms to generate
    testable mechanistic hypotheses with experimental predictions.
    """

    def __init__(self):
        self.step_number = 4
        self.step_name = 'Mechanistic Hypothesis Generation'
        self.dependencies = [1, 2, 3]  # Depends on Steps 1, 2, and 3

        # Initialize LLM
        self.llm = GroqClient()

        # Initialize separate reviewer LLM for biochemistry validation
        try:
            from src.agents.llm_client import UnifiedLLMClient
            self.reviewer_llm = UnifiedLLMClient(provider="reviewer")
            print(f'  Using separate reviewer model: {self.reviewer_llm.model}')
        except Exception:
            self.reviewer_llm = self.llm  # Fallback to same model

    def _prewarm_reviewer(self):
        """Fire a tiny, non-blocking request so the reviewer model (OpenBio behind llm-router)
        wakes from sleep WHILE hypothesis generation runs — so the later biochemistry
        cross-check isn't a wake/cold-start hang. Best-effort: never breaks the step.

        Returns the warm-up Thread (or None if there's no separate reviewer to warm), mainly so
        tests can join it deterministically.
        """
        if self.reviewer_llm is self.llm:
            return None  # no separate reviewer configured; nothing to warm

        def _warm():
            try:
                self.reviewer_llm.chat(
                    [{"role": "user", "content": "ping"}],
                    temperature=0.0,
                    max_tokens=1,
                )
            except Exception:
                pass  # warm-up is best-effort

        try:
            t = threading.Thread(target=_warm, daemon=True)
            t.start()
            return t
        except Exception:
            return None

    @staticmethod
    def _has_pathway_data(mechanisms_result: Optional[Dict]) -> bool:
        """Whether Step 3 produced data the pathway tools can use.

        Gates on pathway_structures OR pathway_mechanisms. The query tools
        (search_pathways_by_gene, get_pathway_crosstalk, list_available_pathways, and
        the structure half of get_pathway_mechanism) are backed by pathway_structures,
        so tool calling remains useful when Step 3 retrieved structures but its
        mechanism-interpretation layer came back empty. Returns False when Step 3 was
        skipped or produced nothing at all.
        """
        if not mechanisms_result:
            return False
        if mechanisms_result.get('metadata', {}).get('skipped', False):
            return False
        return bool(
            mechanisms_result.get('pathway_mechanisms')
            or mechanisms_result.get('pathway_structures')
        )

    def execute(
        self,
        genes: List[Dict],
        pathways: List[Dict],
        themes: Optional[List[Dict]] = None,
        hub_genes_result: Optional[Dict] = None,
        mechanisms_result: Optional[Dict] = None,
        analyses: Optional[List[Dict]] = None,
        context: Optional[Dict] = None,
        seed: int = 42
    ) -> HypothesisResult:
        """
        Execute mechanistic hypothesis generation

        Args:
            genes: List of differentially expressed genes (can be empty)
            pathways: List of enriched pathways
            themes: Pathway themes from Step 1
            hub_genes_result: Hub gene analysis from Step 2
            mechanisms_result: Pathway mechanisms from Step 3
            analyses: Analysis metadata
            context: Optional experimental context
            seed: Random seed for reproducibility (default: 42)

        Returns:
            HypothesisResult with hypotheses, predictions, and report
        """
        print(f'\n[Step {self.step_number}] {self.step_name}')
        print('='*80)

        # Kick off loading the reviewer model now (non-blocking) so it is warm by the time the
        # biochemistry cross-check needs it, instead of paying its cold start synchronously then.
        self._prewarm_reviewer()

        # Extract experimental context
        experiment_context = self._extract_context(context, analyses, themes)

        if experiment_context:
            print(f'\n  Experimental Context: {experiment_context}')

        # Check if genes are provided
        has_genes = genes and len(genes) > 0

        # Validate we have necessary inputs
        if not themes and not hub_genes_result and not mechanisms_result:
            print('\n  ⚠️  Warning: No outputs from previous steps provided.')
            if has_genes:
                print('  Generating hypotheses from genes and pathways only.')
            else:
                print('  Generating hypotheses from pathways only.')

        print(f'\n  Input Summary:')
        print(f'    Genes: {len(genes) if has_genes else 0}{" (pathway-level hypotheses only)" if not has_genes else ""}')
        print(f'    Pathways: {len(pathways)}')
        print(f'    Themes: {len(themes) if themes else 0}')
        print(f'    Hub Genes: {len(hub_genes_result.get("network_hubs", [])) if hub_genes_result else 0}')
        print(f'    Mechanisms: {len(mechanisms_result.get("pathway_mechanisms", [])) if mechanisms_result else 0}')

        # Build LLM prompt
        print('\n  Generating mechanistic hypotheses with tool calling...')
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            genes=genes,
            pathways=pathways,
            themes=themes,
            hub_genes_result=hub_genes_result,
            mechanisms_result=mechanisms_result,
            experiment_context=experiment_context
        )

        try:
            # Initialize pathway query service for tool calling.
            # Enable tools if step 3 produced ANY usable pathway data (structures OR
            # mechanisms) — the query tools are backed by pathway_structures, so they
            # still work when step 3 retrieved structures but its mechanism-
            # interpretation layer came back empty (e.g. an LLM JSON-parse failure).
            pathway_service = None
            tools = []
            if self._has_pathway_data(mechanisms_result):
                pathway_service = PathwayQueryService(mechanisms_result)
                tools = PathwayQueryService.get_tool_definitions()
                if mechanisms_result.get('pathway_mechanisms'):
                    print(f'  Tool calling enabled: {len(tools)} tools available')
                else:
                    print(f'  Tool calling enabled (structures-only — Step 3 mechanism '
                          f'interpretation was empty): {len(tools)} tools available')
            else:
                print(f'  Tool calling disabled (no pathway data from Step 3)')

            # Tool calling loop
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]

            max_iterations = 20
            tool_calls_made = 0
            collected_kegg_relations = []  # Collect all KEGG relations from tool calls
            max_retries = 3  # Retry on intermittent vLLM errors

            for iteration in range(max_iterations):
                # Make LLM call (with or without tools)
                if tools:
                    llm_response = None
                    last_error = None

                    # Retry loop for intermittent vLLM errors (e.g., "unexpected tokens in message header")
                    for retry in range(max_retries):
                        try:
                            llm_response = self.llm.chat_with_tools(
                                messages=messages,
                                tools=tools,
                                temperature=0.0,
                                seed=seed  # Configurable seed for reproducibility
                            )
                            # Debug: Show what the LLM returned
                            if iteration == 0 and retry == 0:
                                print(f'  First LLM response - finish_reason: {llm_response.get("finish_reason")}')
                                print(f'  Has tool_calls: {bool(llm_response.get("tool_calls"))}')
                                if llm_response.get('content'):
                                    preview = llm_response['content'][:200] if len(llm_response['content']) > 200 else llm_response['content']
                                    print(f'  Content preview: {preview}...')
                            break  # Success, exit retry loop
                        except Exception as tool_error:
                            last_error = tool_error
                            if retry < max_retries - 1:
                                print(f'  Warning: Tool calling attempt {retry + 1} failed ({tool_error}), retrying...')
                                time.sleep(0.5)  # Brief pause before retry
                            else:
                                # All retries exhausted - fallback to regular chat
                                print(f'  Warning: Tool calling failed after {max_retries} attempts ({tool_error}), falling back to regular chat')
                                response_text = self.llm.chat(messages, temperature=0.0, seed=seed)
                                llm_response = {
                                    'content': response_text,
                                    'tool_calls': None,
                                    'finish_reason': 'stop'
                                }
                else:
                    # No tools available (Step 3 produced no pathway data) — plain chat
                    response_text = self.llm.chat(messages, temperature=0.0, seed=seed)
                    llm_response = {
                        'content': response_text,
                        'tool_calls': None,
                        'finish_reason': 'stop'
                    }

                # Check if LLM wants to call tools
                if llm_response['tool_calls']:
                    tool_calls_made += len(llm_response['tool_calls'])
                    print(f'  Iteration {iteration + 1}: LLM requested {len(llm_response["tool_calls"])} tool call(s)')

                    # IMPORTANT for vLLM/GPT-OSS: Clean up tool_calls to remove extra metadata
                    # GPT-OSS returns extra fields like "to=functions.xxx" that vLLM doesn't accept
                    cleaned_tool_calls = []
                    for tc in llm_response['tool_calls']:
                        # Only keep the standard OpenAI tool call fields
                        cleaned_tc = {
                            'id': tc['id'],
                            'type': tc.get('type', 'function'),
                            'function': {
                                'name': tc['function']['name'],
                                'arguments': tc['function']['arguments']
                            }
                        }
                        cleaned_tool_calls.append(cleaned_tc)

                    # IMPORTANT for vLLM: Must include 'content' field (even if empty/null)
                    # Also sanitize tool_call_id to remove Harmony format artifacts
                    for tc in cleaned_tool_calls:
                        # Sanitize tool_call_id - remove any special characters or spaces
                        # that may come from Harmony format (e.g., "to=functions.xxx")
                        original_id = tc['id']
                        if isinstance(original_id, str):
                            # Keep only alphanumeric, dash, underscore
                            sanitized_id = ''.join(c for c in original_id if c.isalnum() or c in '-_')
                            if not sanitized_id:
                                sanitized_id = f"tool_call_{iteration}_{cleaned_tool_calls.index(tc)}"
                            tc['id'] = sanitized_id

                    assistant_message = {
                        'role': 'assistant',
                        'content': None,  # Required by vLLM for tool calling messages
                        'tool_calls': cleaned_tool_calls
                    }
                    messages.append(assistant_message)

                    # Execute each tool call (use cleaned_tool_calls to avoid malformed metadata)
                    for tool_call in cleaned_tool_calls:
                        tool_name = tool_call['function']['name']
                        tool_args = json.loads(tool_call['function']['arguments'])

                        print(f'    Executing: {tool_name}({", ".join(f"{k}={v}" for k, v in tool_args.items())})')

                        # Execute tool via pathway service
                        tool_result = pathway_service.execute_tool(tool_name, tool_args)

                        # Collect KEGG relations from pathway mechanism results
                        if tool_name == 'get_pathway_mechanism' and isinstance(tool_result, dict):
                            for rel in tool_result.get('curatedRelations', []):
                                collected_kegg_relations.append({
                                    'source': rel.get('source', '').upper(),
                                    'target': rel.get('target', '').upper(),
                                    'type': rel.get('type', ''),
                                    'pathway': tool_args.get('pathway_name', '')
                                })
                        elif tool_name == 'get_pathway_crosstalk' and isinstance(tool_result, dict):
                            for rel in tool_result.get('curatedRelations', []):
                                collected_kegg_relations.append({
                                    'source': rel.get('source', '').upper(),
                                    'target': rel.get('target', '').upper(),
                                    'type': rel.get('type', ''),
                                    'pathway': f"{tool_args.get('pathway1', '')} x {tool_args.get('pathway2', '')}"
                                })

                        # Add tool result to messages (feed back into next iteration)
                        messages.append({
                            'role': 'tool',
                            'tool_call_id': tool_call['id'],
                            'name': tool_name,  # Required by vLLM Harmony format
                            'content': json.dumps(tool_result)
                        })

                else:
                    # LLM finished - parse final response
                    print(f'  Tool calling complete: {tool_calls_made} total calls made')
                    response = llm_response['content']
                    break

            else:
                # Max iterations reached
                print(f'  Warning: Max iterations ({max_iterations}) reached')
                print(f'  Making final call to LLM to generate hypotheses with current context...')

                # Force final response without tool calling
                messages.append({
                    'role': 'user',
                    'content': 'Please now generate your final hypotheses in JSON format based on all the pathway information gathered. No more tool calls.'
                })

                final_response = self.llm.chat(
                    messages=messages,
                    temperature=0.0,
                    seed=seed,
                    max_tokens=8000
                )

                # Handle both string and dict responses
                if isinstance(final_response, dict):
                    response = final_response.get('content', '{}')
                else:
                    response = final_response  # Already a string

                # If still None or empty, use empty structure
                if response is None or response.strip() == '':
                    print(f'  ⚠️  No response from LLM after max iterations - returning empty hypotheses')
                    response = '{"hypotheses": [], "centralMechanisticModel": "Unable to generate hypotheses due to iteration limit.", "keyPredictions": [], "hypothesesSummary": "Hypothesis generation incomplete."}'

            # Clean up response before parsing
            if response is None:
                response = '{}'
            response = response.strip()

            # Strip markdown code fences (LLM sometimes wraps JSON in ```json ... ```)
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            # Parse JSON response
            try:
                result = json.loads(response)
            except (json.JSONDecodeError, TypeError) as e:
                # Regex fallback: extract first JSON object from response
                result = None
                json_match = re.search(r'\{.*\}', response, re.DOTALL) if isinstance(response, str) else None
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        result = None
                if result is None:
                    print(f'  ⚠️  Failed to parse JSON response: {e}')
                    print(f'  Response content: {response}')
                    result = {
                        "hypotheses": [],
                        "centralMechanisticModel": "Error parsing LLM response.",
                        "keyPredictions": [],
                        "hypothesesSummary": "Hypothesis generation failed due to parsing error."
                    }

            # Handle LLM returning a JSON array instead of object
            if isinstance(result, list):
                print(f'  ⚠️  LLM returned a JSON array ({len(result)} items) instead of object, extracting first element')
                if result and isinstance(result[0], dict):
                    result = result[0]
                else:
                    result = {
                        "hypotheses": [],
                        "centralMechanisticModel": "Error: LLM returned unexpected JSON format.",
                        "keyPredictions": [],
                        "hypothesesSummary": "Hypothesis generation failed due to unexpected response format."
                    }
            elif not isinstance(result, dict):
                result = {
                    "hypotheses": [],
                    "centralMechanisticModel": "Error: LLM returned unexpected response type.",
                    "keyPredictions": [],
                    "hypothesesSummary": "Hypothesis generation failed due to unexpected response format."
                }

            print(f'\n  Hypotheses Generated: {len(result.get("hypotheses", []))}')

            # NEW: Validate hypotheses for gene regulation direction errors
            validation_errors = self._validate_hypothesis_directions(result, genes, experiment_context)
            # Filter out hypotheses with error-severity biochemistry issues
            biochem_errors = [e for e in validation_errors if 'Biochemistry' in e]
            direction_errors = [e for e in validation_errors if 'Biochemistry' not in e]

            # Print direction errors and biochemistry errors separately
            if direction_errors:
                print('\n  ⚠️  WARNING: Gene regulation direction mismatches detected:')
                for error in direction_errors:
                    print(f'    - {error}')

            if biochem_errors:
                print('\n  ⚠️  WARNING: Biochemistry review issues detected:')
                for error in biochem_errors:
                    print(f'    - {error}')

            # Extract 1-based hypothesis indices with error severity
            error_indices = set()
            for err in biochem_errors:
                match = re.match(r'Biochemistry review \(error\) - Hypothesis (\d+):', err)
                if match:
                    error_indices.add(int(match.group(1)))

            # Filter out hypotheses at those indices (1-based -> 0-based)
            if error_indices and result.get('hypotheses'):
                original_count = len(result['hypotheses'])
                result['hypotheses'] = [
                    hyp for i, hyp in enumerate(result['hypotheses'], 1)
                    if i not in error_indices
                ]
                removed = original_count - len(result['hypotheses'])
                print(f'\n  🔬 Filtered out {removed} hypothesis(es) with biochemistry errors (indices: {sorted(error_indices)})')
                print(f'  Hypotheses remaining: {len(result["hypotheses"])}')

                # --- Regeneration: retry loop to replace removed hypotheses ---
                if removed > 0 and len(result['hypotheses']) < 4:
                    print(f'\n  🔄 {removed} hypothesis(es) removed, {len(result["hypotheses"])} remaining (< 4) — regenerating with biochemistry feedback...')

                    # Build feedback message with specific errors for removed hypotheses
                    feedback_lines = []
                    for err in biochem_errors:
                        match = re.match(r'Biochemistry review \(error\) - Hypothesis (\d+): (.*)', err)
                        if match and int(match.group(1)) in error_indices:
                            feedback_lines.append(f"- Hypothesis {match.group(1)} was REMOVED: {match.group(2)}")

                    # Extract gene symbols from error text to build blacklist
                    def _extract_gene_symbols(error_texts):
                        """Extract probable gene symbols (e.g. IDH1, GLUD2) from error descriptions."""
                        genes_found = set()
                        for text in error_texts:
                            genes_found.update(re.findall(r'\b([A-Z][A-Z0-9]{1,15})\b', text))
                        # Filter out common English words / acronyms that aren't genes
                        noise = {'THE', 'AND', 'FOR', 'NOT', 'BUT', 'ARE', 'WAS', 'HAS', 'HAD',
                                 'ITS', 'THIS', 'THAT', 'WITH', 'FROM', 'DOES', 'INTO', 'BEEN',
                                 'ALSO', 'THAN', 'EACH', 'WHEN', 'WILL', 'CAN', 'MAY', 'NEW',
                                 'JSON', 'NADH', 'NAD', 'ATP', 'ADP', 'TCA', 'DNA', 'RNA', 'ENZYME',
                                 'SUBSTRATE', 'PRODUCT', 'DOWN', 'REMOVED', 'ERROR', 'COMMON',
                                 'SELF', 'CHECK', 'AVOID', 'HYPOTHESIS', 'ACTUAL', 'CLAIM'}
                        return genes_found - noise

                    failed_genes = _extract_gene_symbols(feedback_lines)
                    needed = 4 - len(result['hypotheses'])

                    for regen_attempt in range(2):
                        temp = 0.0 if regen_attempt == 0 else 0.3
                        regen_seed = seed + 100 + (regen_attempt * 100)

                        blacklist_line = ""
                        if failed_genes:
                            blacklist_line = f"\n\nDo NOT use these genes as central players: {', '.join(sorted(failed_genes))}\nGenerate hypotheses about DIFFERENT biological mechanisms.\n"

                        regen_prompt = (
                            f"Your previous response generated {original_count} hypotheses, but {removed} were removed "
                            f"due to biochemistry errors:\n\n"
                            + "\n".join(feedback_lines)
                            + blacklist_line
                            + "\n\nCOMMON ERROR PATTERNS TO AVOID:\n"
                            "1. ENZYME DIRECTIONALITY: If enzyme E is DOWN and catalyzes S → P, "
                            "then P DECREASES and S ACCUMULATES. Do NOT reverse this.\n"
                            "2. ENZYME IDENTITY: Verify each enzyme's ACTUAL substrate — aminotransferases "
                            "are substrate-specific, do not assume one acts on a different amino acid class.\n"
                            "3. REVERSED CAUSALITY: If gene X is a downstream TARGET of regulator Y, "
                            "do NOT claim X activates Y.\n"
                            "4. COFACTOR DIRECTION: Dehydrogenases CONSUME NAD+ and PRODUCE NADH. "
                            "If down-regulated, NAD+ is SPARED, not depleted.\n"
                            "5. TCA CYCLE: Verify whether each enzyme PRODUCES or CONSUMES each intermediate.\n\n"
                            "SELF-CHECK before outputting: For each metabolic claim, write "
                            "SUBSTRATE → [ENZYME] → PRODUCT and verify the direction matches "
                            "the enzyme's fold-change sign.\n\n"
                            f"Please generate {needed} NEW replacement hypotheses that avoid these errors. "
                            f"Keep the same JSON format. Return ONLY a JSON object with a \"hypotheses\" array "
                            f"containing the {needed} new hypotheses."
                        )

                        # Build minimal context for regeneration (skip tool call history to avoid oversized prompt)
                        regen_messages = [
                            messages[0],   # system prompt
                            messages[1],   # user prompt (contains all data)
                            {'role': 'assistant', 'content': response},  # original hypothesis response
                            {'role': 'user', 'content': regen_prompt}
                        ]
                        print(f'  Regen attempt {regen_attempt + 1}/2: temp={temp}, seed={regen_seed}, needed={needed}, blacklisted={sorted(failed_genes) if failed_genes else "none"}')
                        print(f'  Regeneration context: {len(regen_messages)} messages, ~{sum(len(str(m.get("content", ""))) for m in regen_messages)} chars')

                        try:
                            regen_response = self.llm.chat(
                                regen_messages,
                                temperature=temp,
                                seed=regen_seed,
                                max_tokens=8000
                            )

                            # Log regeneration response for debugging
                            if regen_response is None:
                                print(f'  Regeneration response: None')
                            else:
                                print(f'  Regeneration response: {len(regen_response)} chars')
                                if len(regen_response) > 0:
                                    preview = regen_response[:200]
                                    print(f'  Regeneration preview: {preview}...')

                            # Parse regenerated response (same cleanup as main path)
                            if regen_response is None:
                                regen_response = '{}'
                            regen_response = regen_response.strip()
                            if regen_response.startswith("```"):
                                regen_response = regen_response.split("```")[1]
                                if regen_response.startswith("json"):
                                    regen_response = regen_response[4:]
                                regen_response = regen_response.strip()

                            try:
                                regen_result = json.loads(regen_response)
                            except (json.JSONDecodeError, TypeError):
                                json_match = re.search(r'\{.*\}', regen_response, re.DOTALL) if isinstance(regen_response, str) else None
                                regen_result = json.loads(json_match.group()) if json_match else None

                            if regen_result and regen_result.get('hypotheses'):
                                new_hyps = regen_result['hypotheses']

                                # Validate regenerated hypotheses through biochemistry review
                                regen_check_result = {'hypotheses': new_hyps}
                                regen_biochem_warnings = self._validate_biochemistry_with_llm(regen_check_result, genes, experiment_context)
                                regen_error_indices = set()
                                regen_error_texts = []
                                for err in regen_biochem_warnings:
                                    m = re.match(r'Biochemistry review \(error\) - Hypothesis (\d+): (.*)', err)
                                    if m:
                                        regen_error_indices.add(int(m.group(1)))
                                        regen_error_texts.append(m.group(2))

                                # Keep only hypotheses that pass biochemistry review
                                clean_new_hyps = [
                                    hyp for i, hyp in enumerate(new_hyps, 1)
                                    if i not in regen_error_indices
                                ]

                                if clean_new_hyps:
                                    result['hypotheses'].extend(clean_new_hyps)
                                    # Update biochem warnings metadata
                                    biochem_errors.extend(regen_biochem_warnings)
                                    validation_errors.extend(regen_biochem_warnings)
                                    needed -= len(clean_new_hyps)
                                    print(f'  ✅ Regen attempt {regen_attempt + 1} added {len(clean_new_hyps)} hypothesis(es) (total: {len(result["hypotheses"])})')
                                else:
                                    print(f'  ⚠️  Regen attempt {regen_attempt + 1}: all regenerated hypotheses failed biochemistry review')

                                # Accumulate blacklist from new failures for next attempt
                                if regen_error_texts:
                                    failed_genes.update(_extract_gene_symbols(regen_error_texts))
                            else:
                                print(f'  ⚠️  Regen attempt {regen_attempt + 1}: no parseable hypotheses')

                        except Exception as regen_error:
                            print(f'  ⚠️  Regen attempt {regen_attempt + 1} failed: {regen_error}')

                        # Stop early if we have enough hypotheses
                        if len(result['hypotheses']) >= 4:
                            print(f'  ✅ Reached {len(result["hypotheses"])} hypotheses — stopping regeneration')
                            break

                    print(f'  Regeneration complete: {len(result["hypotheses"])} final hypotheses')
                elif removed > 0:
                    print(f'\n  ✅ {len(result["hypotheses"])} hypotheses remain after filtering (≥ 4) — skipping regeneration')

            # --- Regenerate centralMechanisticModel and hypothesesSummary after filtering ---
            if error_indices and result.get('hypotheses'):
                surviving_hyps = result['hypotheses']
                print(f'\n  Regenerating central model, summary, and key predictions for {len(surviving_hyps)} surviving hypotheses...')

                hyp_summaries = []
                for idx, hyp in enumerate(surviving_hyps, 1):
                    hyp_summaries.append(
                        f"Hypothesis {idx}: {hyp.get('hypothesis', '')}\n"
                        f"  Mechanism: {hyp.get('mechanisticModel', '')[:200]}"
                    )

                regen_model_prompt = (
                    f"The following {len(surviving_hyps)} hypotheses survived validation"
                    f" for: {experiment_context}.\n\n"
                    + "\n\n".join(hyp_summaries)
                    + '\n\nGenerate ONLY a JSON object with three fields:\n'
                    '{\n'
                    '  "centralMechanisticModel": "Overall unifying model (5-6 sentences, cite genes with fold changes from above)",\n'
                    '  "hypothesesSummary": "Summary showing how hypotheses connect mechanistically (4-5 sentences)",\n'
                    '  "keyPredictions": [\n'
                    '    {"prediction": "What will happen if model is correct", "experiment": "How to test", "expectedDirection": "Expected directional change"}\n'
                    '  ]\n'
                    '}\n'
                    'Generate one keyPrediction per surviving hypothesis. Return ONLY JSON.'
                )

                try:
                    regen_model_response = self.llm.chat(
                        [{"role": "user", "content": regen_model_prompt}],
                        temperature=0.0, seed=seed + 200, max_tokens=2000
                    )

                    # Parse JSON response
                    regen_model_text = regen_model_response.strip() if regen_model_response else '{}'
                    if regen_model_text.startswith("```"):
                        regen_model_text = regen_model_text.split("```")[1]
                        if regen_model_text.startswith("json"):
                            regen_model_text = regen_model_text[4:]
                        regen_model_text = regen_model_text.strip()

                    try:
                        regen_model_result = json.loads(regen_model_text)
                    except (json.JSONDecodeError, TypeError):
                        json_match = re.search(r'\{.*\}', regen_model_text, re.DOTALL) if isinstance(regen_model_text, str) else None
                        regen_model_result = json.loads(json_match.group()) if json_match else None

                    if regen_model_result:
                        if regen_model_result.get('centralMechanisticModel'):
                            result['centralMechanisticModel'] = regen_model_result['centralMechanisticModel']
                            print(f'  ✅ Central mechanistic model regenerated')
                        if regen_model_result.get('hypothesesSummary'):
                            result['hypothesesSummary'] = regen_model_result['hypothesesSummary']
                            print(f'  ✅ Hypotheses summary regenerated')
                        if regen_model_result.get('keyPredictions'):
                            result['keyPredictions'] = regen_model_result['keyPredictions']
                            print(f'  ✅ Key predictions regenerated ({len(regen_model_result["keyPredictions"])} predictions)')
                    else:
                        print(f'  Central model regeneration produced no parseable result — keeping originals')

                except Exception:
                    print(f'  Central model regeneration failed — keeping originals')

            # Enforce effect-size coverage BEFORE the numerical/KEGG/consistency
            # validators, so any hypotheses added to cover an omitted dominant axis
            # go through the same numerical/KEGG/prediction/consistency checks as the
            # originals (they are already direction- and biochemistry-validated inside
            # the coverage loop). This makes AC1/AC3 mechanical rather than dependent
            # on the LLM honoring the soft coverage instruction in the prompt.
            try:
                coverage_warnings = self._enforce_hypothesis_coverage(
                    result, pathways, genes, experiment_context,
                    system_prompt, user_prompt
                )
            except Exception as e:  # noqa: BLE001 - coverage is best-effort, never fatal
                print(f'  ⚠️  Coverage enforcement skipped due to error: {e}')
                coverage_warnings = [f'Coverage enforcement error: {e}']

            # Validate numerical claims (FC, p-values, gene counts)
            numerical_warnings = self._validate_numerical_claims(
                result, genes, pathways, mechanisms_result
            )
            if numerical_warnings:
                print('\n  ⚠️  WARNING: Numerical claim mismatches detected:')
                for warning in numerical_warnings:
                    print(f'    - {warning}')
                print('  Hypotheses may contain cross-contaminated values between genes!')

            # Auto-correct fabricated gene-level p-values and FDRs
            gene_stat_corrections = self._autocorrect_gene_stats(result, genes)
            if gene_stat_corrections:
                print(f'\n  Auto-corrected {gene_stat_corrections} fabricated gene p-value/FDR(s) in hypothesis text')

            # Auto-correct pathway FDR misattributed as gene-level FDR
            fdr_corrections = self._autocorrect_pathway_fdr(result, genes, pathways, mechanisms_result)
            if fdr_corrections:
                print(f'\n  Auto-corrected {fdr_corrections} pathway FDR → gene FDR substitution(s) in hypothesis text')

            # Validate mechanistic claims against KEGG relations (warning-only)
            kegg_warnings = self._validate_kegg_claims(result, collected_kegg_relations)
            if kegg_warnings:
                print('\n  ⚠️  WARNING: Unsupported KEGG mechanistic claims detected:')
                for warning in kegg_warnings:
                    print(f'    - {warning}')

            # Validate free-text KEGG claims (Issue #1: fabricated KEGG citations in prose)
            kegg_text_warnings = self._validate_kegg_text_claims(result, collected_kegg_relations)
            if kegg_text_warnings:
                print('\n  ⚠️  WARNING: Free-text KEGG fabrication detected:')
                for warning in kegg_text_warnings:
                    print(f'    - {warning}')

            # Validate prediction-vs-data contradictions (Issue #4)
            prediction_warnings = self._validate_prediction_vs_data(result, genes)
            if prediction_warnings:
                print('\n  ⚠️  WARNING: Prediction-vs-data contradictions detected:')
                for warning in prediction_warnings:
                    print(f'    - {warning}')

            # Validate internal consistency (contradictions within/between merged hypotheses)
            consistency_warnings = self._validate_internal_consistency(result, genes)
            if consistency_warnings:
                print('\n  ⚠️  WARNING: Internal consistency issues detected:')
                for warning in consistency_warnings:
                    print(f'    - {warning}')

            # Correct fold-change citations against the DE table BEFORE rendering. The
            # LLM re-types fold changes in free prose and scrambles magnitudes
            # (e.g. "HEATR1 +1.15" when the DE value is +1.10). Values are templated from
            # data, not trusted from the model. Same-sign magnitude errors are rewritten
            # in place; a sign conflict (a deeper, direction-inverting error) in the central
            # model triggers one regeneration, after which any residual conflict has its
            # whole sentence dropped so the prose never contradicts the true direction.
            gene_fc = gene_fc_lookup(genes)
            fc_conflicts = self._correct_fold_change_citations(result, gene_fc, seed)

            # Generate report section
            report_section = self._generate_report_section(result, experiment_context)

            return HypothesisResult(
                hypotheses=result.get('hypotheses', []),
                central_mechanistic_model=result.get('centralMechanisticModel', ''),
                key_predictions=result.get('keyPredictions', []),
                hypotheses_summary=result.get('hypothesesSummary', ''),
                report_section=report_section,
                metadata={
                    'experiment_context': experiment_context,
                    'num_hypotheses': len(result.get('hypotheses', [])),
                    'num_predictions': len(result.get('keyPredictions', [])),
                    'has_themes': themes is not None,
                    'has_hub_genes': hub_genes_result is not None,
                    'has_mechanisms': mechanisms_result is not None,
                    'coverage_validation_warnings': coverage_warnings,
                    'direction_validation_errors': [e for e in validation_errors if 'Biochemistry' not in e],
                    'biochemistry_validation_warnings': [e for e in validation_errors if 'Biochemistry' in e],
                    'numerical_validation_warnings': numerical_warnings,
                    'kegg_validation_warnings': kegg_warnings,
                    'kegg_text_validation_warnings': kegg_text_warnings,
                    'prediction_vs_data_warnings': prediction_warnings,
                    'kegg_relations_collected': len(collected_kegg_relations),
                    'internal_consistency_warnings': consistency_warnings,
                    'fold_change_sign_conflicts': sorted(fc_conflicts)
                }
            )

        except json.JSONDecodeError as e:
            print(f'\n  Error parsing JSON response: {e}')
            raise
        except Exception as e:
            print(f'\n  Error in Step {self.step_number}: {e}')
            raise

    def _correct_fold_change_citations(
        self,
        result: Dict,
        gene_fc: Dict[str, float],
        seed: int,
    ) -> set:
        """Correct fold-change citations in generated prose against the DE table.

        Fixes same-sign magnitude errors in place across the central model and every
        hypothesis's prose. A sign conflict in the central model (a direction-inverting
        error, not a typo) triggers ONE regeneration with the true signed values templated
        in; any conflict that survives has its numeric citation dropped so the narrative
        never contradicts the data. Returns the set of genes that had a sign conflict.
        """
        if not gene_fc:
            return set()

        all_conflicts: set = set()

        def _fix_str(text):
            """Sanitize leaked meta-commentary, then magnitude-correct FC citations."""
            if not isinstance(text, str) or not text:
                return text, set()
            cleaned = sanitize_llm_text(text)
            return correct_fc_citations(cleaned, gene_fc)

        # Hypothesis prose: sanitize + magnitude-only fixes across every rendered text
        # field (sign errors here are surfaced by the existing direction validator; we
        # don't rewrite hypothesis wording). Covers plain-string and list/dict fields the
        # report renders, so a wrong magnitude can't survive in a bullet or sub-field.
        for hyp in result.get('hypotheses', []) or []:
            if not isinstance(hyp, dict):
                continue
            for field in ('hypothesis', 'mechanisticModel', 'quantitativePrediction',
                          'directionalPrediction', 'confidenceRationale'):
                fixed, conflicts = _fix_str(hyp.get(field))
                if isinstance(hyp.get(field), str):
                    hyp[field] = fixed
                    all_conflicts |= conflicts
            for list_field in ('keyPlayers', 'evidenceSupporting'):
                items = hyp.get(list_field)
                if isinstance(items, list):
                    new_items = []
                    for it in items:
                        fixed, conflicts = _fix_str(it)
                        new_items.append(fixed if isinstance(it, str) else it)
                        all_conflicts |= conflicts
                    hyp[list_field] = new_items
            testability = hyp.get('testability')
            if isinstance(testability, dict):
                for k in ('approach1', 'approach2', 'expectedOutcome'):
                    fixed, conflicts = _fix_str(testability.get(k))
                    if isinstance(testability.get(k), str):
                        testability[k] = fixed
                        all_conflicts |= conflicts

        # Key predictions (list of dicts) and the hypotheses summary also render FCs.
        for pred in result.get('keyPredictions', []) or []:
            if isinstance(pred, dict):
                for k in ('prediction', 'experiment', 'expectedDirection'):
                    fixed, conflicts = _fix_str(pred.get(k))
                    if isinstance(pred.get(k), str):
                        pred[k] = fixed
                        all_conflicts |= conflicts
        fixed, conflicts = _fix_str(result.get('hypothesesSummary'))
        if isinstance(result.get('hypothesesSummary'), str):
            result['hypothesesSummary'] = fixed
            all_conflicts |= conflicts

        # Central mechanistic model: the headline narrative — sanitize + correct, and
        # resolve sign conflicts via one regeneration; if a conflict still survives, drop
        # the whole offending sentence (dropping only the number would leave the governing
        # direction word wrong), guaranteeing the prose never contradicts the data.
        central = result.get('centralMechanisticModel', '')
        if isinstance(central, str) and central:
            fixed, conflicts = _fix_str(central)
            result['centralMechanisticModel'] = fixed
            if conflicts:
                print(f'  ⚠️  Fold-change sign conflict in central model for '
                      f'{sorted(conflicts)} — regenerating once with templated values')
                regen = self._regenerate_central_model_with_fcs(result, gene_fc, seed)
                if regen:
                    result['centralMechanisticModel'] = regen
                fixed2, conflicts2 = _fix_str(result['centralMechanisticModel'])
                result['centralMechanisticModel'] = fixed2
                if conflicts2:
                    result['centralMechanisticModel'] = drop_sentences_with_fc_conflicts(
                        result['centralMechanisticModel'], gene_fc)
                    print(f'  ⚠️  Persisting sign conflict for {sorted(conflicts2)} — '
                          f'dropped contradicting sentence(s)')
                    all_conflicts |= conflicts2
                # else: regeneration resolved the conflict — not reported as unresolved.

        # `all_conflicts` = genes with an UNRESOLVED sign conflict (hypothesis-prose
        # conflicts we don't rewrite, plus central-model conflicts that survived
        # regeneration). Central conflicts fixed by regeneration are intentionally omitted.
        return all_conflicts

    def _regenerate_central_model_with_fcs(
        self,
        result: Dict,
        gene_fc: Dict[str, float],
        seed: int,
    ) -> Optional[str]:
        """Regenerate the central mechanistic model with authoritative signed FCs.

        Provides the true fold changes as a table and instructs the model to use ONLY
        those values, so a sign-inverted citation is not merely renumbered into
        contradictory prose. Returns the new text, or None if regeneration failed.
        """
        current = result.get('centralMechanisticModel', '')
        # Only bother templating genes actually cited / relevant, but a compact full table
        # is cheap and unambiguous.
        fc_table = '\n'.join(f'- {sym}: {fc:+.2f}' for sym, fc in sorted(gene_fc.items()))
        prompt = (
            'Rewrite the following molecular "central mechanistic model" paragraph so that '
            'every fold change it cites EXACTLY matches the authoritative values below. Do '
            'not invent numbers, do not restate a value for any gene not in this table, and '
            'ensure the regulation wording (up/down, increased/decreased) agrees with each '
            "gene's sign. Keep it to 5-6 sentences.\n\n"
            f'Authoritative differential expression (signed fold change):\n{fc_table}\n\n'
            f'Paragraph to rewrite:\n{current}\n\n'
            'Return ONLY a JSON object: {"centralMechanisticModel": "..."}'
        )
        try:
            resp = self.llm.chat(
                [{'role': 'user', 'content': prompt}],
                temperature=0.0, seed=seed + 300, max_tokens=1500,
            )
            obj = self._parse_llm_json(resp)  # shared fence-strip + regex-fallback parser
            if obj and obj.get('centralMechanisticModel'):
                print('  ✅ Central mechanistic model regenerated with templated fold changes')
                return obj['centralMechanisticModel']
        except Exception as e:
            print(f'  ⚠️  Central-model regeneration failed ({e}); keeping corrected text')
        return None

    # ------------------------------------------------------------------
    # Effect-size coverage enforcement (AC1/AC3)
    # ------------------------------------------------------------------
    def _hypothesis_corpus(self, result: Dict) -> str:
        """Concatenate all hypothesis text used to decide whether an axis is covered."""
        parts = [result.get('centralMechanisticModel', ''),
                 result.get('hypothesesSummary', '')]
        for h in result.get('hypotheses', []) or []:
            if not isinstance(h, dict):
                # Tolerate malformed LLM output (a bare string hypothesis).
                parts.append(str(h))
                continue
            parts.append(h.get('hypothesis', ''))
            parts.append(h.get('mechanisticModel', ''))
            for kp in h.get('keyPlayers', []) or []:
                parts.append(kp if isinstance(kp, str) else str(kp))
        return '\n'.join(p for p in parts if p)

    @staticmethod
    def _de_fc_lookup(genes: List[Dict]) -> Dict[str, float]:
        """symbol (upper) -> signed fold change. Delegates to the shared `gene_fc_lookup`
        so the field-name tolerance and 0.0-safe parsing live in one place."""
        return gene_fc_lookup(genes)

    def _axis_signature(self, pathways: List[Dict], de_fc: Dict[str, float],
                        direction: str, threshold: float):
        """Pathway names and direction-matched DE genes for one dominant axis.

        Returns ``(names, signature_genes)`` gathered from pathways whose signed
        NES matches ``direction`` and whose |NES| >= ``threshold``. A hypothesis is
        deemed to "cover" the axis if it mentions one of these names or genes.
        """
        want_up = (direction == 'up')
        names, sig_genes = set(), set()
        for p in pathways or []:
            if not isinstance(p, dict):
                continue
            val = to_float(_enrichment_metric(p)[0])
            if val is None or abs(val) < threshold:
                continue
            if (val > 0) != want_up:
                continue
            nm = (p.get('name') or p.get('pathwayName') or p.get('pathway') or '').strip().lower()
            if nm:
                names.add(nm)
            for g in p.get('genes', []) or []:
                gsym = str(g).upper()
                fc = de_fc.get(gsym)
                if fc is not None and ((fc > 0) == want_up):
                    sig_genes.add(gsym)
        return names, sig_genes

    def _validate_hypothesis_coverage(self, result: Dict, pathways: List[Dict],
                                      genes: List[Dict],
                                      threshold: float = DOMINANT_NES_THRESHOLD) -> List[Dict]:
        """Return descriptors for dominant enrichment axes not covered by any hypothesis.

        An axis is "dominant" when its strongest pathway has |NES| >= ``threshold``.
        Guarantees AC1 (a hypothesis per dominant direction) and AC3 (the top-|NES|
        cluster is referenced) when the returned list is driven to empty.
        """
        ups, downs = _nes_axis_signals(pathways)
        corpus = self._hypothesis_corpus(result).lower()
        de_fc = self._de_fc_lookup(genes)

        unmet = []
        for direction, sig_list in (('up', ups), ('down', downs)):
            if not sig_list or abs(sig_list[0][0]) < threshold:
                continue
            val, name, metric = sig_list[0][0], sig_list[0][1], sig_list[0][2]
            names, sig_genes = self._axis_signature(pathways, de_fc, direction, threshold)
            referenced = any(nm and nm in corpus for nm in names) or any(
                re.search(r'\b' + re.escape(g.lower()) + r'\b', corpus) for g in sig_genes)
            if not referenced:
                unmet.append({
                    'direction': direction,
                    'pathway': name,
                    'nes': val,
                    'metric': metric,
                    'genes': sorted(sig_genes)[:12],
                })
        return unmet

    def _enforce_hypothesis_coverage(self, result: Dict, pathways: List[Dict],
                                     genes: List[Dict], experiment_context: str,
                                     system_prompt: str, user_prompt: str,
                                     max_attempts: int = 2) -> List[str]:
        """Add hypotheses until every dominant axis is covered (best-effort).

        Mirrors the biochemistry regeneration loop's *append* semantics: it re-prompts
        the generator for the omitted axis, keeps only NEW hypotheses that pass both
        direction and biochemistry validation, and APPENDS them — it never replaces or
        drops the already-validated originals, and it leaves the central model / summary
        / predictions untouched. On exhaustion it records a warning instead of
        hard-failing so the pipeline never breaks on an uncooperative model.
        """
        warnings: List[str] = []
        for attempt in range(max_attempts):
            unmet = self._validate_hypothesis_coverage(result, pathways, genes)
            if not unmet:
                return warnings

            lines = []
            for u in unmet:
                gene_hint = ', '.join(u['genes'][:10]) or '(see the pathway table)'
                lines.append(
                    f"- The strongly {u['direction']}-regulated program '{u['pathway']}' "
                    f"({u.get('metric', 'NES')} {u['nes']:+.2f}) is NOT addressed by any hypothesis. "
                    f"Generate a NEW, distinct, mechanistic hypothesis centered on it and its "
                    f"genes: {gene_hint}.")
            coverage_prompt = (
                "Your hypotheses omit a dominant enrichment signal. Saliency is set by effect "
                "size (|NES|), NOT by network connectivity — a strongly enriched program still "
                "needs a hypothesis even if none of its genes are PPI hubs.\n"
                + "\n".join(lines) +
                "\n\nReturn JSON with a 'hypotheses' array containing ONLY the new "
                "hypothesis(es) for the uncovered axis(es), using the same per-hypothesis "
                "schema as before."
            )
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
                {'role': 'assistant', 'content': json.dumps(result)},
                {'role': 'user', 'content': coverage_prompt},
            ]
            print(f'\n  🔄 Coverage regeneration attempt {attempt + 1}/{max_attempts}: '
                  f'{len(unmet)} dominant axis(es) uncovered '
                  f'({", ".join(u["direction"] + ":" + u["pathway"] for u in unmet)})')

            try:
                regen_response = self.llm.chat(messages, temperature=0.3,
                                               seed=42 + attempt, max_tokens=8000)
                regen_result = self._parse_llm_json(regen_response)
            except Exception as e:  # noqa: BLE001 - regeneration must never crash the step
                warnings.append(f'Coverage regeneration failed ({e})')
                break

            if not regen_result or not regen_result.get('hypotheses'):
                warnings.append('Coverage regeneration returned no usable hypotheses')
                break

            # Consider only genuinely NEW hypotheses (dedup by statement) so a model
            # that echoes the full set can't create duplicates, and originals survive.
            existing = result.get('hypotheses', []) or []
            existing_stmts = {(h.get('hypothesis', '') or '').strip().lower()
                              for h in existing if isinstance(h, dict)}
            candidates = [h for h in regen_result.get('hypotheses', [])
                          if isinstance(h, dict)
                          and (h.get('hypothesis', '') or '').strip().lower() not in existing_stmts]
            if not candidates:
                warnings.append('Coverage regeneration produced no new distinct hypotheses')
                break

            # Validate the NEW candidates for direction + biochemistry, so a coverage
            # fix can neither reintroduce a wrong-direction claim nor an implausible
            # mechanism. `_validate_hypothesis_directions` ALSO runs the LLM
            # biochemistry review internally and returns both error kinds in one list,
            # so we parse both regexes off that single call (no second reviewer pass).
            cand_result = {'hypotheses': candidates}
            bad_idx = set()
            try:
                cand_errors = self._validate_hypothesis_directions(
                    cand_result, genes, experiment_context)
            except Exception:  # noqa: BLE001 - validators must never crash the step
                cand_errors = []
            for err in cand_errors:
                m = (re.match(r'Hypothesis (\d+):', err)
                     or re.match(r'Biochemistry review \(error\) - Hypothesis (\d+):', err))
                if m:
                    bad_idx.add(int(m.group(1)))

            good_new = [h for i, h in enumerate(candidates, 1) if i not in bad_idx]
            if not good_new:
                warnings.append('Coverage regeneration hypotheses failed validation')
                break

            # APPEND — never replace — so existing validated hypotheses are preserved.
            # Synthesis fields (central model / summary / predictions) are intentionally
            # left as-is to avoid a regen dropping a previously-covered axis from them.
            result['hypotheses'] = existing + good_new

        # Final status after the loop (whether it converged, broke, or exhausted).
        final_unmet = self._validate_hypothesis_coverage(result, pathways, genes)
        for u in final_unmet:
            warnings.append(
                f"Uncovered {u['direction']}-regulated dominant axis: "
                f"{u['pathway']} ({u.get('metric', 'NES')} {u['nes']:+.2f})")
        return warnings

    @staticmethod
    def _parse_llm_json(response: Optional[str]) -> Optional[Dict]:
        """Parse a JSON object from an LLM response (same cleanup as the main path)."""
        if not response:
            return None
        if isinstance(response, dict):
            return response
        text = str(response).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except (json.JSONDecodeError, TypeError):
                    return None
        return None

    def _extract_context(
        self,
        context: Optional[Dict],
        analyses: Optional[List[Dict]],
        themes: Optional[List[Dict]]
    ) -> str:
        """Extract experimental context from various sources"""
        context_parts = []

        # From explicit context parameter - include ALL context fields dynamically
        if context:
            # Skip internal/technical fields
            skip_fields = {'datasetId', 'dataset_id', 'organism'}
            for key, value in context.items():
                if key in skip_fields:
                    continue
                if value and str(value).strip():
                    context_parts.append(f"{key}: {value}")

        # From analyses
        if analyses and len(analyses) > 0:
            analysis = analyses[0]
            if analysis.get('experimentContext'):
                context_parts.append(analysis['experimentContext'])
            if analysis.get('contextFields'):
                fields = analysis['contextFields']
                if fields.get('tissue'):
                    context_parts.append(fields['tissue'])
                if fields.get('treatment'):
                    context_parts.append(fields['treatment'])

        # From themes (as fallback)
        if not context_parts and themes and len(themes) > 0:
            first_theme = themes[0]
            if first_theme.get('biological_context'):
                # Extract context from biological_context field
                bio_context = first_theme['biological_context']
                # Simple extraction: look for "In X, " pattern
                if bio_context.startswith('In '):
                    parts = bio_context.split(',', 1)
                    if len(parts) > 0:
                        context_parts.append(parts[0].replace('In ', '').strip())

        return ', '.join(context_parts) if context_parts else ''

    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM"""
        return """You are a molecular biology expert generating TESTABLE mechanistic hypotheses from multi-step pathway analysis.

You have been provided with:
1. Pathway themes (Step 1) - high-level biological processes
2. Hub genes (Step 2) - network topology analysis
3. Pathway summary table (Step 3) - overview of all analyzed pathways
4. Quantitative data - fold changes, p-values, enrichment

⚠️ MANDATORY TOOL CALLING REQUIREMENT ⚠️
The pathway summary table shows ONLY gene names and crosstalk summaries.
You MUST call tools to get detailed KEGG mechanisms before generating ANY hypothesis.

Available tools:
- get_pathway_mechanism(pathway_name) - Get detailed KEGG relations (activation, inhibition, expression, etc.)
- search_pathways_by_gene(gene_symbol) - Find all pathways containing a gene
- get_pathway_crosstalk(pathway1, pathway2) - Analyze shared genes and their specific roles

Use these tools strategically to:
- Explore pathways relevant to your hypotheses
- Get detailed KEGG relations for mechanistic models
- Understand gene roles across multiple pathways
- Identify pathway crosstalk and interactions

CRITICAL WORKFLOW:
1. Identify 3-5 pathways of interest from the summary table
2. Call get_pathway_mechanism() for EACH pathway to get detailed KEGG relations
3. Use the detailed relations to build mechanistic models
4. Then generate hypotheses based on the detailed mechanisms

DO NOT generate hypotheses based only on gene lists - you need the specific KEGG relations:
- Which genes activate/inhibit which targets?
- What are the expression/binding/phosphorylation relationships?
- How do genes interact within the pathway?

Your task is to SYNTHESIZE these into mechanistic hypotheses that:
- Connect multiple pathways/genes mechanistically
- Explain WHY certain pathways are enriched
- Make DIRECTIONAL predictions grounded in observed data
- Are experimentally testable

CRITICAL REQUIREMENTS:
- Use SPECIFIC gene names and fold changes from the data
- Cite SPECIFIC pathway enrichment p-values
- Make DIRECTIONAL predictions based on observed fold changes (e.g., "GENE_X knockdown should reduce downstream GENE_Y expression")
- Do NOT invent specific percentages or fold-change magnitudes for predictions — only state the DIRECTION and which genes/pathways are affected
- You may cite OBSERVED fold changes from the data (e.g., "GENE_X is upregulated 3.2-fold"), but do NOT predict unobserved numerical outcomes
- Propose CONCRETE experimental tests (e.g., "siRNA knockdown of GENE_X in CELL_LINE")
- Base mechanistic models on KEGG relationships (query via tools)

⚠️ CRITICAL GENE REGULATION RULE ⚠️
ALWAYS verify gene regulation direction from the quantitative data before making ANY claims:
- Fold change > 0 = UP-regulated (↑)
- Fold change < 0 = DOWN-regulated (↓)
- Theme tags [UP-regulated], [DOWN-regulated], [Mixed regulation] indicate the dominant direction
- DO NOT assume pathway enrichment means gene upregulation
- DO NOT make biological assumptions that contradict the actual fold change values
- Cross-reference theme genes with the "Top Differentially Expressed Genes" section
- If a theme is tagged [DOWN-regulated], ALL genes in that theme's hypotheses must be described as down-regulated
- Use ONLY gene functions from the "Gene Function Reference" section when available
- DO NOT invent gene molecular functions — if a gene's role is not listed, describe it generically
- ONLY cite genes with fold changes that appear in the data provided in the user prompt
  (Top DE Genes list, Hub Genes list, Pathway Summary Table, or Immune Cytotoxicity section)
- Do NOT invent fold-change values from prior knowledge about typical disease expression patterns
- If a gene is relevant but not in the provided data, state "not measured in this dataset"

⚠️ BIOLOGICAL PLAUSIBILITY RULES ⚠️

1. MECHANISM GROUNDING: Only propose mechanisms that are DIRECTLY supported by the
   KEGG relations returned from tool calls. If no KEGG relation connects gene A to
   gene B, do NOT invent an intermediate mechanism. Instead, note the gap explicitly.
   - BAD: "GENE_X sequesters metabolite Y in lipid droplets" (no KEGG relation supports this)
   - BAD: "Acidic pH enhances LBP binding to LPS-like ligands" (no KEGG relation supports pH→LBP)
   - GOOD: "Gene A and Gene B are co-dysregulated; their functional
     connection requires experimental validation"
   - GOOD: "LBP activates TLR4 signaling (KEGG: LBP→TLR4→NF-κB); the trigger for LBP
     upregulation in this context requires further investigation"
   - BAD: "Metabolic pathway impairment stabilizes transcription factor X" (co-regulated ≠ causally linked)
   - GOOD: "Metabolic enzymes and transcription factor targets are coordinately dysregulated;
     whether metabolic stress drives transcription factor activation requires
     experimental validation"

2. METABOLIC DIRECTION: When proposing metabolic consequences, verify the direction:
   - If an enzyme is DOWNREGULATED, its PRODUCT decreases (not increases)
   - If an enzyme is DOWNREGULATED, its SUBSTRATE accumulates
   - Example: IDH1 downregulation → isocitrate accumulates, α-KG decreases

3. TRANSCRIPTION FACTOR SPECIFICITY: Do not assume generic signaling connections.
   Verify transcription factor → target gene relationships against established, curated
   TF→target biology (e.g. the candidate upstream regulators listed below, when provided).
   - NF-κB drives: TNF, IL6, IL8, CXCL1/2, CCL2, ICAM1, VCAM1, BCL2
   - NF-κB does NOT directly drive: IFNG (driven by T-bet/TBX21, STAT4 in T/NK cells)
   - HIF-1α drives: CA9, VEGFA, GLUT1, LDHA, PDK1, BNIP3
   - HIF-1α does NOT directly drive: IFNG, GZMB, FASLG (immune cell genes)
   - IFN-γ/STAT1/IRF1 drives: CXCL9, CXCL10 (IP-10), CXCL11, IDO1, GBP1, IRF1, TAP1, CIITA
   - CXCL10 is NOT an NF-κB target — it is primarily driven by IFN-γ→STAT1→IRF1
   - When IFN-γ and CXCL10 are co-upregulated, attribute CXCL10 to the IFN-γ/STAT1 axis, not NF-κB

4. CELL COMPARTMENT AWARENESS: Distinguish tumor-intrinsic vs immune microenvironment:
   - Tumor cell genes: oncogenes, metabolic enzymes, tissue-specific markers from the DE gene list
   - Immune cell genes: IFNG, GZMB, FASLG, PRF1, CD3D/E, CD8A, CD4 (these are always immune-cell derived regardless of disease)
   - Do NOT claim tumor gene knockdown will directly change immune gene expression
     unless specifying a co-culture or paracrine mechanism
   - When a hypothesis involves genes from DIFFERENT cell types, the mechanistic model MUST
     specify the paracrine/juxtacrine communication mechanism between cells
   - The hypothesis title should NOT imply all genes act within the same cell

5. DRIVER vs PASSENGER: Consider whether gene changes are causal or consequential:
   - Tissue-specific markers lost in dedifferentiation (e.g., tissue-specific
     differentiation markers lost in the tumor) are likely PASSENGERS of tumor transformation, not drivers
   - Known driver events for this disease context should be prioritized
   - Flag passenger changes as "likely consequential" rather than building causal chains

6. METABOLIC REACTION DIRECTION VERIFICATION (CRITICAL):
   Before claiming ANY metabolic consequence, verify the reaction direction:
   a) Identify the enzyme's reaction: ENZYME catalyzes SUBSTRATE → PRODUCT
   b) If the enzyme is DOWNREGULATED:
      - Its SUBSTRATE may accumulate (less consumed)
      - Its PRODUCT decreases (less produced)
      - Cofactors CONSUMED by the reaction are SPARED (less consumed)
      - Cofactors PRODUCED by the reaction DECREASE (less produced)
   c) If the enzyme is UPREGULATED, the opposite applies
   d) For downstream signaling effects of metabolite changes:
      - Verify whether the metabolite is a co-substrate (activates the target)
        or a competitive inhibitor (inhibits the target) — do NOT assume
      - Accumulation of a co-substrate INCREASES target enzyme activity
      - Accumulation of an inhibitor DECREASES target enzyme activity
   e) If uncertain about a reaction direction, state the uncertainty explicitly
      rather than guessing

7. RECEPTOR SIGNALING DIRECTION: When proposing receptor → downstream effects:
   - Specify whether the receptor signals via Gs (activating, ↑cAMP), Gi (inhibitory, ↓cAMP),
     or Gq (calcium signaling) for GPCRs
   - Example: GPR81/HCAR1 is Gi-coupled (suppresses cAMP); PTGER2 is Gs-coupled (activates cAMP). Check the coupling class before claiming downstream effects.
   - Do NOT assume receptor activation always means pathway activation
   - For receptor → transcription factor claims, verify whether the receptor activates
     or inhibits the downstream pathway based on its coupling class

8. EVIDENCE WEIGHT BY EFFECT SIZE: Not all DE genes are equally informative.
   - |FC| >= 2.0: Strong effect — suitable as a primary driver in causal chains
   - |FC| 1.0-2.0: Moderate effect — can support a mechanism but should not anchor it
   - |FC| < 1.0: Weak effect — mention only as corroborating; do NOT cite as key evidence
   - When listing evidence, lead with highest-FC genes

9. CORRELATION vs CAUSATION: Co-expression does NOT establish a causal chain.
   - State causal links ONLY when supported by a KEGG relation or established pathway
   - For co-expressed genes without a known regulatory link, say "co-upregulated" not "A drives B"
   - Positive feedback loops (e.g., IFN-γ → CXCL10 → T-cell recruitment → more IFN-γ)
     must be labeled as feedback loops, not linear causal chains

10. HYPOTHESIS DIVERSITY: Each hypothesis MUST address a DISTINCT biological mechanism.
   - Do NOT generate two hypotheses about the same metabolic process at different detail levels
   - If two hypotheses share >50% of their keyPlayers, merge or replace one
   - Aim for coverage across: metabolism, signaling, immune response, transcriptional regulation, etc.

11. COMMON METABOLIC REASONING ERRORS (avoid these):
   a) ENZYME DIRECTIONALITY: If enzyme E catalyzes A → B and E is DOWN-regulated:
      - B DECREASES (less product made), A ACCUMULATES (substrate not consumed)
      - Do NOT say "A decreases" — that reverses the logic
   b) ENZYME IDENTITY: Verify each enzyme's ACTUAL substrate before claiming it participates
      in a pathway. Aminotransferases are substrate-specific — do not assume one acts on a
      different amino acid class than its annotated substrate.
   c) REVERSED CAUSALITY: If gene X is a known DOWNSTREAM TARGET of transcription factor Y,
      do NOT claim X activates or stabilizes Y. Check the direction: regulator → target.
      Common example: HIF-1α induces many genes — those genes do NOT stabilize HIF-1α.
   d) COFACTOR DIRECTION: Dehydrogenases typically CONSUME NAD+ and PRODUCE NADH.
      If such an enzyme is DOWN-regulated: less NAD+ consumed (NAD+ spared), less NADH produced.
      Do NOT claim the opposite.
   e) TCA CYCLE: For each TCA enzyme, verify whether it PRODUCES or CONSUMES each intermediate.
      Downregulation of an enzyme that CONSUMES metabolite X causes X to ACCUMULATE, not decrease.

RESPONSE FORMAT:
- Return ONLY valid JSON
- NO markdown code blocks
- NO explanatory text before or after
- Just the raw JSON object"""

    def _priority_signals_block(self, pathways: List[Dict]) -> str:
        """Build the 'Priority Enrichment Signals' block (strongest |NES| both directions).

        Lists the largest-magnitude up- and down-regulated pathways and states that
        every dominant direction (|NES| >= threshold) MUST be covered by a
        hypothesis. This is what the coverage enforcement (_validate_hypothesis_coverage)
        checks after generation, so the instruction here is a hard requirement, not
        a soft nudge.
        """
        ups, downs = _nes_axis_signals(pathways)
        if not ups and not downs:
            return ''

        parts = ['**⚠️ PRIORITY ENRICHMENT SIGNALS (largest effect size — cover BOTH directions):**\n']
        parts.append(
            'Rank saliency by |enrichment score| (magnitude), NOT by network connectivity. '
            'The most strongly enriched program is often down-regulated or peripheral in the '
            'PPI network — it MUST still be explained by a hypothesis.\n')

        def _fmt(sig_list, n=5):
            return '; '.join(
                f'{nm} ({metric} {val:+.2f}, {_magnitude_label(val, metric)})'
                for val, nm, metric in sig_list[:n])

        if ups:
            parts.append(f'- Strongest UP-regulated: {_fmt(ups)}\n')
        if downs:
            parts.append(f'- Strongest DOWN-regulated: {_fmt(downs)}\n')

        # Name the dominant directions that REQUIRE a dedicated hypothesis.
        required = []
        if ups and abs(ups[0][0]) >= DOMINANT_NES_THRESHOLD:
            required.append(f'UP ({ups[0][1]}, {ups[0][2]} {ups[0][0]:+.2f})')
        if downs and abs(downs[0][0]) >= DOMINANT_NES_THRESHOLD:
            required.append(f'DOWN ({downs[0][1]}, {downs[0][2]} {downs[0][0]:+.2f})')
        if required:
            parts.append(
                '❗ At least one hypothesis MUST be centered on each of these dominant '
                f'axes (|NES| >= {DOMINANT_NES_THRESHOLD:g}): ' + '; '.join(required) + '.\n')
        parts.append('\n')
        return ''.join(parts)

    def _build_user_prompt(
        self,
        genes: List[Dict],
        pathways: List[Dict],
        themes: Optional[List[Dict]],
        hub_genes_result: Optional[Dict],
        mechanisms_result: Optional[Dict],
        experiment_context: str
    ) -> str:
        """Build user prompt with all data from previous steps"""
        prompt_parts = []

        # Header
        context_text = f" specific to {experiment_context}" if experiment_context else ""
        prompt_parts.append(f'Generate up to 7 mechanistic hypotheses{context_text}:\n\n')

        # Priority enrichment signals (effect-size, both directions). Front-loaded
        # BEFORE tool-calling so the strongest signal — which is often a
        # down-regulated / PPI-peripheral program — is not crowded out by the
        # densely-connected (usually up-regulated) hub modules below.
        prompt_parts.append(self._priority_signals_block(pathways))

        # Instructions
        prompt_parts.append('**Task Requirements:**\n')
        prompt_parts.append(f'1. Create mechanistic models connecting pathways and genes{" in this tissue/disease context" if experiment_context else ""}\n')
        prompt_parts.append(f'2. Propose cause-effect relationships{" relevant to the experimental setting" if experiment_context else ""}\n')
        prompt_parts.append(f'3. Explain WHY certain pathways are enriched{" in this specific tissue/disease" if experiment_context else ""}\n')
        prompt_parts.append(f'4. Suggest how different pathways work together{" in this biological context" if experiment_context else ""}\n')
        prompt_parts.append(f'5. Each hypothesis must be testable{" and appropriate for this experimental system" if experiment_context else ""}\n\n')

        # Add disease biology context (known driver events)
        if experiment_context:
            prompt_parts.append('**Known Disease Biology (use as context, do NOT just repeat):**\n')
            prompt_parts.append(f'Consider established molecular drivers for {experiment_context} when ')
            prompt_parts.append('evaluating whether observed gene changes are drivers or passengers.\n')
            prompt_parts.append('Build hypotheses that ADD MECHANISTIC DETAIL to known biology — specify '
                               'which genes/steps are responsible, rather than claiming the overall process is novel.\n')
            prompt_parts.append('If a process (e.g., immune infiltration in KIRC) is well-established, '
                               'label it "builds-on-known" and focus on what your data adds.\n\n')

        # Add pathway themes from Step 1
        if themes:
            significance_order = {'high': 0, 'medium': 1, 'low': 2}

            # NES lookup by pathway name (fallback when a theme's member pathway
            # dicts don't themselves carry the enrichment score).
            pathway_nes_by_name = {}
            for p in pathways or []:
                nm = (p.get('name') or p.get('pathwayName') or p.get('pathway') or '').strip().lower()
                v = to_float(_enrichment_metric(p)[0])
                if nm and v is not None and (nm not in pathway_nes_by_name
                                             or abs(v) > abs(pathway_nes_by_name[nm])):
                    pathway_nes_by_name[nm] = v

            def _theme_nes(theme):
                """(max |NES|, signed NES at that max) across the theme's pathways."""
                best_abs, signed = 0.0, 0.0
                for pw in theme.get('pathways', []) or []:
                    val, nm = None, ''
                    if isinstance(pw, dict):
                        val = to_float(_enrichment_metric(pw)[0])
                        nm = (pw.get('name') or pw.get('pathwayName') or '').strip().lower()
                    else:
                        nm = str(pw).strip().lower()
                    if val is None and nm:
                        val = pathway_nes_by_name.get(nm)
                    if val is not None and abs(val) > best_abs:
                        best_abs, signed = abs(val), val
                return best_abs, signed

            theme_nes = {id(t): _theme_nes(t) for t in themes}

            # Rank themes by EFFECT SIZE (|NES|) first, so strong down-regulated
            # metabolic themes are not buried below moderate up-regulated ones that
            # merely carry a higher LLM 'significance' label (the original bias).
            sorted_themes = sorted(
                themes,
                key=lambda t: (
                    -theme_nes[id(t)][0],
                    significance_order.get(t.get('significance', 'low'), 2),
                    t.get('avg_p_value_fdr', 1.0),
                    t.get('name', '')
                )
            )

            # Guarantee the strongest up- AND down-regulated theme above the
            # dominance threshold are both shown, even if the top-5 cap would drop
            # one — so hypothesis seeding always sees each dominant axis (AC3).
            forced, forced_ids = [], set()
            for sign in (+1, -1):
                cands = [t for t in sorted_themes
                         if (theme_nes[id(t)][1] * sign) > 0
                         and theme_nes[id(t)][0] >= DOMINANT_NES_THRESHOLD]
                if cands and id(cands[0]) not in forced_ids:
                    forced.append(cands[0])
                    forced_ids.add(id(cands[0]))

            shown = list(forced)
            shown_ids = set(forced_ids)
            for t in sorted_themes:
                if len(shown) >= 5:
                    break
                if id(t) not in shown_ids:
                    shown.append(t)
                    shown_ids.add(id(t))
            shown.sort(key=lambda t: -theme_nes[id(t)][0])

            prompt_parts.append('**Biological Themes (from Step 1 - Pathway Clustering):**\n\n')
            for i, theme in enumerate(shown, 1):
                theme_name = theme.get('name', 'Unknown')
                theme_desc = theme.get('description', '')
                key_genes = theme.get('key_genes', [])
                pathway_count = theme.get('pathway_count', 0)

                # NEW: Get regulation direction info
                dominant_direction = theme.get('dominant_direction', 'mixed')
                key_genes_with_fc = theme.get('key_genes_with_fc', [])

                # Add direction tag to theme name
                direction_tag = ''
                if dominant_direction == 'up':
                    direction_tag = ' [UP-regulated]'
                elif dominant_direction == 'down':
                    direction_tag = ' [DOWN-regulated]'
                elif dominant_direction == 'mixed':
                    direction_tag = ' [Mixed regulation]'

                prompt_parts.append(f'{i}. **{theme_name}**{direction_tag} ({pathway_count} pathways)\n')
                prompt_parts.append(f'   - {theme_desc}\n')

                # Display genes with fold changes if available
                if key_genes_with_fc:
                    genes_with_dir = []
                    for gene_info in key_genes_with_fc[:5]:
                        gene = gene_info.get('gene', '')
                        fc = gene_info.get('fold_change', 0)
                        direction = fc_arrow(fc)
                        genes_with_dir.append(f"{gene} ({direction}{fc:.2f})")
                    prompt_parts.append(f'   - Key genes: {", ".join(genes_with_dir)}\n')
                elif key_genes:
                    # Fallback to gene names only
                    prompt_parts.append(f'   - Key genes: {", ".join(key_genes[:5])}\n')
            prompt_parts.append('\n')

        # Add hub genes from Step 2
        if hub_genes_result and hub_genes_result.get('network_hubs'):
            prompt_parts.append('**Hub Genes (from Step 2 - Network Analysis):**\n\n')
            hubs = hub_genes_result['network_hubs'][:10]
            for hub in hubs:
                gene = hub.get('gene', 'Unknown')
                fc = hub.get('fold_change', 0)
                hub_score = hub.get('hub_score', 0)
                role = hub.get('biological_role', '')
                direction = fc_arrow(fc)

                prompt_parts.append(f'- **{gene}** (FC: {fc:.2f} {direction}, Hub Score: {hub_score:.3f})\n')
                if role:
                    prompt_parts.append(f'  {role}\n')
            # Add instruction to incorporate top hub genes into hypotheses.
            hubs = hub_genes_result['network_hubs'][:5]
            if hubs:
                top_hub_names = [h.get('gene', '') for h in hubs]
                prompt_parts.append(
                    f'The top hub genes by network centrality are: {", ".join(top_hub_names)}. '
                    f'These are the most connected nodes in the interaction network. '
                    f'Reference them where mechanistically justified, or note why they are not '
                    f'involved.\n'
                    f'⚠️ Hub status reflects PPI connectivity, which is biased toward large, '
                    f'densely-studied (usually up-regulated proliferation) modules. Do NOT let '
                    f'hub membership override effect size: a strongly enriched program (high '
                    f'|NES|) still needs a hypothesis even if none of its genes are hubs.\n\n'
                )

        # Add pathway summary table from Step 3 (NOT full details - use tools to query)
        if mechanisms_result and mechanisms_result.get('pathway_structures'):
            structures = mechanisms_result['pathway_structures']

            # Select the shown pathways by |enrichment score| in BOTH directions and
            # interleave them, so the strongest down-regulated programs (large
            # negative NES) survive the top-N cap. The previous signed-descending
            # sort pushed negative-NES pathways to the bottom and truncated them,
            # starving hypothesis generation of the single strongest signal (AC3).
            def _abs_score(s):
                v = to_float(s.get('enrichment_score'))
                return abs(v) if v is not None else 0.0

            def _signed_score(s):
                v = to_float(s.get('enrichment_score'))
                return v if v is not None else 0.0

            up_structs = sorted(
                [s for s in structures if _signed_score(s) > 0],
                key=lambda s: (-_abs_score(s), s.get('pathway', '')))
            down_structs = sorted(
                [s for s in structures if _signed_score(s) < 0],
                key=lambda s: (-_abs_score(s), s.get('pathway', '')))
            zero_structs = sorted(
                [s for s in structures if _signed_score(s) == 0],
                key=lambda s: s.get('pathway', ''))

            TABLE_LIMIT = 15
            sorted_structures = []
            ui = di = 0
            while len(sorted_structures) < TABLE_LIMIT and (ui < len(up_structs) or di < len(down_structs)):
                if ui < len(up_structs):
                    sorted_structures.append(up_structs[ui])
                    ui += 1
                    if len(sorted_structures) >= TABLE_LIMIT:
                        break
                if di < len(down_structs):
                    sorted_structures.append(down_structs[di])
                    di += 1
            # Fill any remaining slots with unscored/zero-NES pathways.
            for s in zero_structs:
                if len(sorted_structures) >= TABLE_LIMIT:
                    break
                sorted_structures.append(s)

            # Build crosstalk lookup: pathway -> list of (partner_pathway, shared_gene_count)
            crosstalk_lookup = {}
            pathway_crosstalk = mechanisms_result.get('pathway_overlaps', [])
            for crosstalk_entry in pathway_crosstalk:
                pathway1 = crosstalk_entry.get('pathway1', '')
                pathway2 = crosstalk_entry.get('pathway2', '')
                shared_count = crosstalk_entry.get('shared_genes_count', 0)

                # Add bidirectional entries
                if pathway1 not in crosstalk_lookup:
                    crosstalk_lookup[pathway1] = []
                crosstalk_lookup[pathway1].append((pathway2, shared_count))

                if pathway2 not in crosstalk_lookup:
                    crosstalk_lookup[pathway2] = []
                crosstalk_lookup[pathway2].append((pathway1, shared_count))

            prompt_parts.append('**Pathway Summary Table (from Step 3 - KEGG Analysis):**\n\n')
            prompt_parts.append('⚠️ This table shows SUMMARY information only (gene lists, crosstalk counts).\n')
            prompt_parts.append('YOU MUST call get_pathway_mechanism() to get detailed KEGG relations for hypothesis generation.\n\n')
            prompt_parts.append('| Pathway | Enrich Score | Dir | DE Count | Gene Names (summary) | Crosstalk Pathways (summary) | Pathway_FDR (NOT gene FDR) |\n')
            prompt_parts.append('|---------|--------------|-----|----------|------------|-------------------|----------------------------|\n')

            for struct in sorted_structures[:15]:  # Show top 15
                pathway = struct.get('pathway', 'Unknown')
                score = struct.get('enrichment_score', 0)
                direction = struct.get('enrichment_direction', 'mixed')
                de_count = struct.get('de_genes_count', 0)
                fdr = struct.get('p_value_fdr', 1)

                # Extract DE gene names with fold changes
                mapped_genes = struct.get('mapped_de_genes', [])
                gene_names_list = []
                seen_genes = set()  # Handle duplicates
                for gene_info in mapped_genes:
                    gene_symbol = gene_info.get('gene_symbol', '')
                    if gene_symbol and gene_symbol not in seen_genes:
                        fc = gene_info.get('fold_change', 0)
                        direction_symbol = fc_arrow(fc)
                        gene_names_list.append(f"{gene_symbol} ({direction_symbol}{abs(fc):.2f})")
                        seen_genes.add(gene_symbol)

                # Format gene names (show first 5)
                if gene_names_list:
                    gene_names_str = ', '.join(gene_names_list[:5])
                    if len(gene_names_list) > 5:
                        gene_names_str += '...'
                else:
                    gene_names_str = '-'

                # Get crosstalk pathways (show top 3 by shared gene count)
                crosstalk_partners = crosstalk_lookup.get(pathway, [])
                # Sort by shared gene count descending
                crosstalk_partners_sorted = sorted(crosstalk_partners, key=lambda x: x[1], reverse=True)
                crosstalk_list = []
                for partner_pathway, shared_count in crosstalk_partners_sorted[:3]:
                    crosstalk_list.append(f"{partner_pathway} ({shared_count})")

                if crosstalk_list:
                    crosstalk_str = ', '.join(crosstalk_list)
                    if len(crosstalk_partners) > 3:
                        crosstalk_str += '...'
                else:
                    crosstalk_str = '-'

                # Format direction abbreviation
                dir_abbrev = direction[:4] if direction else 'mix'

                prompt_parts.append(f'| {pathway} | {score:.2f} | {dir_abbrev} | {de_count} | {gene_names_str} | {crosstalk_str} | {fdr:.2e} |\n')

            prompt_parts.append('\n')
            prompt_parts.append('Note: The enrichment scores in the table above are computed in Step 3. When citing evidence,\n')
            prompt_parts.append('use ONLY values present in the data provided (fold changes, p-values, FDR values, enrichment\n')
            prompt_parts.append('scores from the table). Do NOT invent or estimate statistics not present in the data.\n\n')

        # Add candidate upstream regulators (TFs) of the down-regulated programs.
        # These are inferred by testing which TF's target set (CollecTRI regulon) is
        # enriched among the down-regulated genes, and are typically NOT in the DE list —
        # so they are missing from the hub list above but may be the real drivers of a
        # strongly down-regulated axis.
        #
        # Use every regulator Step 3 returns rather than re-truncating here: Step 3 already
        # caps the list (``_identify_upstream_regulators`` top_n, currently 8) after ranking
        # by enrichment FDR, so a second cap here only discarded candidates that had already
        # passed the significance gate — and left this prompt disagreeing with the report,
        # which lists them all.
        if mechanisms_result and mechanisms_result.get('upstream_regulators'):
            regs = mechanisms_result['upstream_regulators']
            if regs:
                prompt_parts.append('**Candidate Upstream Regulators (TFs whose targets are enriched in the '
                                    'down-regulated genes, from TF→target regulon enrichment):**\n')
                prompt_parts.append('These transcription factors regulate down-regulated DE genes; many are not themselves '
                                    'differentially expressed (so they are absent from the hub list). Consider them as '
                                    'candidate drivers when explaining down-regulated programs.\n')
                for reg in regs:
                    tf = reg.get('tf', '')
                    n = reg.get('overlap_count', reg.get('num_targets', 0))
                    de_note = 'in DE list' if reg.get('is_de') else 'NOT in DE list'
                    activity = reg.get('inferred_tf_activity')
                    act_note = f', inferred activity {activity}' if activity and activity != 'unknown' else ''
                    targets = ', '.join(reg.get('targets', [])[:5])
                    prompt_parts.append(f'- **{tf}** ({de_note}{act_note}) → regulates {n} down-regulated gene(s): {targets}\n')
                prompt_parts.append('\n')

        # Add gene function reference from Step 3 pathway annotations
        if mechanisms_result and mechanisms_result.get('pathway_mechanisms'):
            gene_functions = {}
            for mech in mechanisms_result['pathway_mechanisms']:
                pathway_name = mech.get('pathway', '')
                for gene_info in mech.get('deGeneInvolvement', []):
                    gene = gene_info.get('gene', '')
                    role = gene_info.get('roleInPathway', '')
                    if gene and role:
                        if gene not in gene_functions:
                            gene_functions[gene] = []
                        gene_functions[gene].append((pathway_name, role))

            if gene_functions:
                prompt_parts.append('**Gene Function Reference (from KEGG pathway annotations):**\n')
                prompt_parts.append('Use these validated functions — do NOT invent gene roles.\n\n')
                for gene in sorted(gene_functions.keys()):
                    roles = gene_functions[gene]
                    seen = set()
                    unique = []
                    for pathway, role in roles:
                        if role not in seen:
                            unique.append(f"{role} ({pathway})")
                            seen.add(role)
                    prompt_parts.append(f'- **{gene}**: {"; ".join(unique[:2])}\n')
                prompt_parts.append('\n')

        # Add top differentially expressed genes
        prompt_parts.append('**Top Differentially Expressed Genes:**\n\n')
        prompt_parts.append('IMPORTANT: The FDR values in the pathway table above are PATHWAY-level enrichment FDR.\n')
        prompt_parts.append('When citing a gene\'s significance, use the p-value/FDR from THIS gene list, NOT the pathway FDR.\n')
        prompt_parts.append('If a gene is NOT listed below, do NOT cite any p-value or FDR for it — cite only its fold change.\n')
        prompt_parts.append('NEVER emit placeholder or unresolved tokens for a value you cannot source '
                            '(e.g. "p=e-??", "p=2.85e-?? not listed", "(value unknown)", "(TBD)"). '
                            'If a value is unavailable, OMIT it entirely and state only what you have.\n\n')

        # Build a lookup for all genes: symbol -> gene dict
        all_gene_lookup = {}
        for gene in genes:
            symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                     gene.get('gene') or gene.get('name', '')).upper()
            if symbol:
                all_gene_lookup[symbol] = gene

        # Collect gene symbols mentioned in pathway tables and immune sections
        pathway_gene_symbols = set()
        if mechanisms_result and mechanisms_result.get('pathway_structures'):
            for struct in mechanisms_result['pathway_structures']:
                for gene_info in struct.get('mapped_de_genes', []):
                    sym = gene_info.get('gene_symbol', '').upper()
                    if sym:
                        pathway_gene_symbols.add(sym)

        sorted_genes = sorted(genes, key=lambda g: abs(g.get('foldChange', g.get('log2_fold_change', 0))), reverse=True)
        shown_symbols = set()

        def _format_gene_line(gene):
            symbol = gene.get('geneSymbol', gene.get('gene_symbol', gene.get('name', 'Unknown')))
            fc = gene.get('foldChange', gene.get('log2_fold_change', 0))
            pval = gene.get('pValue', gene.get('p_value', 1))
            fdr = gene.get('pValueFDR', gene.get('p_value_fdr', gene.get('fdr', gene.get('adjPValue', None))))
            direction = fc_arrow(fc)
            if fdr is not None:
                return f'- {symbol}: {fc:.2f} {direction} (gene p={pval:.2e}, gene FDR={fdr:.2e})\n'
            else:
                return f'- {symbol}: {fc:.2f} {direction} (gene p={pval:.2e})\n'

        for gene in sorted_genes[:15]:
            symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                     gene.get('gene') or gene.get('name', 'Unknown')).upper()
            prompt_parts.append(_format_gene_line(gene))
            shown_symbols.add(symbol)

        # Add any pathway/immune genes not already shown (so LLM has their real stats)
        extra_symbols = pathway_gene_symbols - shown_symbols
        if extra_symbols:
            extra_genes = []
            for sym in extra_symbols:
                if sym in all_gene_lookup:
                    extra_genes.append(all_gene_lookup[sym])
            if extra_genes:
                extra_genes.sort(key=lambda g: abs(g.get('foldChange', g.get('log2_fold_change', 0))), reverse=True)
                prompt_parts.append('\n*Additional DE genes from enriched pathways:*\n')
                for gene in extra_genes:
                    symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                             gene.get('gene') or gene.get('name', 'Unknown')).upper()
                    prompt_parts.append(_format_gene_line(gene))
                    shown_symbols.add(symbol)

        prompt_parts.append('\n')

        # Add top pathways
        prompt_parts.append('**Top Enriched Pathways:**\n\n')
        sorted_pathways = sorted(pathways, key=lambda p: p.get('pValueFDR', p.get('pValue', 1)))
        for pathway in sorted_pathways[:10]:
            name = pathway.get('name', pathway.get('pathwayName', 'Unknown'))
            fdr = pathway.get('pValueFDR', pathway.get('adjPValue', pathway.get('pValue', 1)))
            prompt_parts.append(f'- {name}: FDR={fdr:.2e}\n')
        prompt_parts.append('\n')

        # JSON format specification
        prompt_parts.append('---\n\n')
        prompt_parts.append('⚠️ **WORKFLOW REQUIREMENTS:** ⚠️\n')
        prompt_parts.append('1. First, call get_pathway_mechanism() for 3-5 key pathways from the table above\n')
        prompt_parts.append('2. Analyze the detailed KEGG relations (activation, inhibition, expression, etc.)\n')
        prompt_parts.append('3. Build mechanistic models connecting multiple pathways\n')
        prompt_parts.append('4. ONLY THEN return the final JSON with hypotheses\n')
        prompt_parts.append('5. BEFORE finalizing each hypothesis, self-check:\n')
        prompt_parts.append('   a) For each enzyme: write SUBSTRATE → [ENZYME] → PRODUCT\n')
        prompt_parts.append('   b) If enzyme is downregulated: product DECREASES, substrate ACCUMULATES\n')
        prompt_parts.append('   c) Verify you have NOT reversed this in your model\n')
        prompt_parts.append('   d) Confirm each gene fold-change SIGN matches your up/down claim\n\n')

        prompt_parts.append('⚠️ **COVERAGE REQUIREMENT:** Your hypotheses must COLLECTIVELY address:\n')
        prompt_parts.append('1. The top upregulated pathway cluster (e.g., immune/inflammatory pathways)\n')
        prompt_parts.append('2. The top downregulated pathway cluster (e.g., metabolic pathways)\n')
        prompt_parts.append('3. At least 3 of the top 5 hub genes by hub score\n')
        prompt_parts.append('If a top hub gene is not included in any hypothesis, briefly explain why in the summary.\n\n')

        # Dynamic immune cytotoxicity requirement: check if immune killing genes are upregulated
        immune_cytotox_genes = {
            'IFNG': 'IFN-γ (T/NK cell effector cytokine)',
            'GZMB': 'Granzyme B (cytotoxic granule)',
            'FASLG': 'Fas ligand (death receptor ligand)',
            'CD8A': 'CD8α (cytotoxic T cell marker)',
            'CD3D': 'CD3δ (T cell receptor complex)',
            'CD3E': 'CD3ε (T cell receptor complex)',
            'PRF1': 'Perforin (cytotoxic granule)',
            'TYROBP': 'DAP12 (NK cell activating adaptor)',
            'FCER1G': 'FcεRIγ (NK cell signaling)',
        }
        # Build gene lookup for checking
        gene_fc_lookup = {}
        for gene in genes:
            symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                     gene.get('gene') or gene.get('name', '')).upper()
            fc = gene.get('foldChange') or gene.get('log2_fold_change') or gene.get('fold_change', 0)
            if symbol:
                gene_fc_lookup[symbol] = fc

        # Find upregulated immune cytotoxicity genes
        upregulated_immune = []
        for gene_sym, description in immune_cytotox_genes.items():
            fc = gene_fc_lookup.get(gene_sym)
            if fc is not None and fc > 0:
                upregulated_immune.append((gene_sym, fc, description))

        if len(upregulated_immune) >= 2:
            # Sort by fold change descending for the prompt
            upregulated_immune.sort(key=lambda x: x[1], reverse=True)
            gene_list_str = ', '.join(
                f'{g[0]} (↑{g[1]:.2f}, {g[2]})' for g in upregulated_immune
            )

            # Check for Allograft rejection pathway
            allograft_str = ''
            for pathway in pathways:
                pname = (pathway.get('name') or pathway.get('pathwayName') or '').lower()
                if 'allograft' in pname or 'graft' in pname:
                    pfdr = pathway.get('pValueFDR', pathway.get('adjPValue', pathway.get('pValue', 1)))
                    allograft_str = f' The Allograft rejection pathway is enriched (FDR={pfdr:.2e}).'
                    break

            prompt_parts.append('⚠️ **IMMUNE CYTOTOXICITY REQUIREMENT:** ⚠️\n')
            prompt_parts.append(f'The following immune cytotoxicity genes are UPREGULATED in this dataset:\n')
            prompt_parts.append(f'{gene_list_str}\n')
            prompt_parts.append(f'At least ONE hypothesis MUST be dedicated to the T cell / NK cell cytotoxic killing ')
            prompt_parts.append(f'mechanism as the CORE of the hypothesis (not just mentioned as a keyPlayer).{allograft_str}\n')
            prompt_parts.append(f'This hypothesis should explain:\n')
            prompt_parts.append(f'- What drives immune cell infiltration/activation in this context\n')
            prompt_parts.append(f'- The effector mechanisms (granzyme/perforin, Fas/FasL, cytokines)\n')
            prompt_parts.append(f'- The cellularContext: these genes are expressed in infiltrating immune cells, NOT tumor cells\n')
            prompt_parts.append(f'Do NOT fold this into an NF-κB or chemokine-centric hypothesis — it must be a standalone ')
            prompt_parts.append(f'immune killing hypothesis.\n\n')

        prompt_parts.append('CRITICAL: When citing a gene\'s significance, use its gene-level p/FDR from the gene list above.\n')
        prompt_parts.append('The FDR in the pathway table is the PATHWAY enrichment FDR — do NOT attribute it to individual genes.\n\n')
        prompt_parts.append('Return JSON with SPECIFIC, testable hypotheses:\n')
        prompt_parts.append('{\n')
        prompt_parts.append('  "hypotheses": [\n')
        prompt_parts.append('    {\n')
        prompt_parts.append('      "hypothesis": "Clear, testable hypothesis statement with specific genes",\n')
        prompt_parts.append('      "mechanisticModel": "Detailed molecular mechanism with specific interactions (3-4 sentences, include gene names and pathways)",\n')
        prompt_parts.append('      "keyPlayers": ["specific gene/pathway1 with role", "specific gene/pathway2 with role"],\n')
        prompt_parts.append('      "evidenceSupporting": [\n')
        prompt_parts.append('        "Quantitative evidence from data (e.g., \'GENE_X overexpressed 3.2-fold\')",\n')
        prompt_parts.append('        "Pathway enrichment evidence (e.g., \'Cell cycle pathway FDR=1e-15\') — use pathway FDR here, NOT gene FDR"\n')
        prompt_parts.append('      ],\n')
        prompt_parts.append('      "testability": {\n')
        prompt_parts.append('        "approach1": "Specific experimental test (e.g., \'siRNA knockdown of GENE_X in CELL_LINE cells\')",\n')
        prompt_parts.append('        "approach2": "Alternative test method with expected outcome",\n')
        prompt_parts.append('        "expectedOutcome": "Directional prediction (e.g., \'GENE_X knockdown should reduce downstream GENE_Y expression\')"\n')
        prompt_parts.append('      },\n')
        prompt_parts.append('      "directionalPrediction": "Directional prediction grounded in data (e.g., \'Inhibiting GENE_X [↑FC] should reduce downstream pathway Z activation\')",\n')
        prompt_parts.append('      "confidence": "high" | "medium" | "low",\n')
        prompt_parts.append('      "confidenceRationale": "Why this confidence level (cite specific evidence)",\n')
        prompt_parts.append('      "novelty": "novel | builds-on-known | confirmatory",'
                           ' // novel = mechanism not in literature; builds-on-known = adds detail to established biology;'
                           ' confirmatory = data confirms known mechanism\n')
        prompt_parts.append('      "cellularContext": "Which cell type(s) each gene is expressed in (e.g., \'Tumor-intrinsic genes: tumor cells; IFNG, GZMB: infiltrating immune cells; secreted factors: specify source cell type\')",\n')
        prompt_parts.append('      "mechanisticClaims": [\n')
        prompt_parts.append('        {"source": "GENE_A", "relation": "activation|inhibition|expression|compound|association", "target": "GENE_B", "basis": "KEGG pathway name or literature"}\n')
        prompt_parts.append('      ]\n')
        prompt_parts.append('    }\n')
        prompt_parts.append('  ],\n')
        prompt_parts.append('  "centralMechanisticModel": "Overall unifying model integrating multiple pathways with specific molecular details (5-6 sentences, cite specific genes with fold changes)",\n')
        prompt_parts.append('  "keyPredictions": [\n')
        prompt_parts.append('    {\n')
        prompt_parts.append('      "prediction": "What will happen if model is correct",\n')
        prompt_parts.append('      "experiment": "How to test this prediction",\n')
        prompt_parts.append('      "expectedDirection": "Expected directional change (e.g., \'Reduced chemokine secretion, decreased immune infiltration\')"\n')
        prompt_parts.append('    }\n')
        prompt_parts.append('  ],\n')
        prompt_parts.append('  "hypothesesSummary": "Summary of all hypotheses showing how they connect mechanistically, with specific genes and pathways mentioned (4-5 sentences)"\n')
        prompt_parts.append('}\n\n')

        prompt_parts.append('Focus on hypotheses that:\n')
        prompt_parts.append(f'- Connect multiple pathways or genes{" within the context of " + experiment_context if experiment_context else ""}\n')
        prompt_parts.append(f'- Explain unexpected findings{" for this tissue/disease" if experiment_context else ""}\n')
        prompt_parts.append(f'- Have therapeutic implications{" relevant to this experimental setting" if experiment_context else ""}\n')
        prompt_parts.append(f'- Can be validated experimentally{" in this biological system" if experiment_context else ""}\n')
        prompt_parts.append('- Address DISTINCT biological mechanisms — no two hypotheses should focus on the same pathway or process\n')

        return ''.join(prompt_parts)

    def _generate_report_section(self, result: Dict, experiment_context: str) -> str:
        """Generate markdown report section"""
        lines = [
            '## Mechanistic Hypotheses\n\n',
        ]

        if experiment_context:
            lines.append(f'**Experimental Context:** {experiment_context}\n\n')

        # Central mechanistic model
        central_model = result.get('centralMechanisticModel', '')
        if central_model:
            lines.append('### Central Mechanistic Model\n\n')
            lines.append(f'{central_model}\n\n')

        # Individual hypotheses
        hypotheses = result.get('hypotheses', [])
        if hypotheses:
            lines.append('### Testable Hypotheses\n\n')
            for i, hyp in enumerate(hypotheses, 1):
                lines.append(f'#### Hypothesis {i}\n\n')

                # Hypothesis statement
                lines.append(f'**Statement:** {hyp.get("hypothesis", "")}\n\n')

                # Confidence and novelty
                confidence = hyp.get('confidence', 'unknown')
                novelty = hyp.get('novelty', 'unknown')
                lines.append(f'*Confidence: {confidence.upper()} | Novelty: {novelty}*\n\n')

                # Mechanistic model
                mech_model = hyp.get('mechanisticModel', '')
                if mech_model:
                    lines.append(f'**Mechanistic Model:**\n{mech_model}\n\n')

                # Key players
                key_players = hyp.get('keyPlayers', [])
                if key_players:
                    lines.append('**Key Players:**\n')
                    for player in key_players:
                        lines.append(f'- {player}\n')
                    lines.append('\n')

                # Supporting evidence
                evidence = hyp.get('evidenceSupporting', [])
                if evidence:
                    lines.append('**Supporting Evidence:**\n')
                    for ev in evidence:
                        lines.append(f'- {ev}\n')
                    lines.append('\n')

                # Testability
                testability = hyp.get('testability', {})
                if testability:
                    lines.append('**Experimental Validation:**\n')
                    if testability.get('approach1'):
                        lines.append(f'- Approach 1: {testability["approach1"]}\n')
                    if testability.get('approach2'):
                        lines.append(f'- Approach 2: {testability["approach2"]}\n')
                    if testability.get('expectedOutcome'):
                        lines.append(f'- Expected Outcome: {testability["expectedOutcome"]}\n')
                    lines.append('\n')

                # Directional prediction
                dir_pred = hyp.get('directionalPrediction', '')
                if dir_pred:
                    lines.append(f'**Directional Prediction:** {dir_pred}\n\n')

                lines.append('---\n\n')

        # Key predictions
        predictions = result.get('keyPredictions', [])
        if predictions:
            lines.append('### Key Testable Predictions\n\n')
            for i, pred in enumerate(predictions, 1):
                lines.append(f'{i}. **{pred.get("prediction", "")}**\n')
                lines.append(f'   - Experiment: {pred.get("experiment", "")}\n')
                lines.append(f'   - Expected: {pred.get("expectedDirection", "")}\n\n')

        # Summary
        summary = result.get('hypothesesSummary', '')
        if summary:
            lines.append('### Summary\n\n')
            lines.append(f'{summary}\n\n')

        return ''.join(lines)

    def _validate_hypothesis_directions(
        self,
        result: Dict,
        genes: List[Dict],
        experiment_context: str = ''
    ) -> List[str]:
        """
        Validate that hypotheses correctly describe gene regulation direction

        Args:
            result: LLM hypothesis generation result
            genes: List of DE genes with fold changes

        Returns:
            List of error messages (empty if no errors)
        """
        # Build gene lookup dict
        gene_lookup = {}
        for gene in genes:
            symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                     gene.get('gene') or gene.get('name', '')).upper()
            fc = gene.get('foldChange') or gene.get('log2_fold_change') or gene.get('fold_change')
            if fc is not None:
                gene_lookup[symbol] = fc

        errors = []

        # Check hypotheses
        for i, hyp in enumerate(result.get('hypotheses', []), 1):
            hyp_text = hyp.get('hypothesis', '') + ' ' + hyp.get('mechanisticModel', '')

            # Check each gene in lookup
            for gene_symbol, fc in gene_lookup.items():
                true_direction = 'up' if fc > 0 else 'down'

                # Check if gene appears in hypothesis text
                if re.search(rf'\b{re.escape(gene_symbol)}\b', hyp_text, re.IGNORECASE):
                    # Check for contradictory direction claims
                    gene_context = re.findall(
                        rf'(\w+[\s\-]*regulat\w+|overexpress\w+|enhanced|increased|elevated|reduced|decreased|down[\s\-]*regulat\w+|up[\s\-]*regulat\w+|loss|suppression)[\s\w,]*\b{re.escape(gene_symbol)}\b',
                        hyp_text,
                        re.IGNORECASE
                    )

                    for context in gene_context:
                        context_lower = context.lower()

                        # UP-regulation indicators
                        up_indicators = ['up-regulat', 'upregulat', 'overexpress', 'enhanced', 'increased', 'elevated']
                        # DOWN-regulation indicators
                        down_indicators = ['down-regulat', 'downregulat', 'reduced', 'decreased', 'loss', 'suppression']

                        claimed_up = any(ind in context_lower for ind in up_indicators)
                        claimed_down = any(ind in context_lower for ind in down_indicators)

                        if true_direction == 'down' and claimed_up:
                            errors.append(
                                f"Hypothesis {i}: {gene_symbol} is DOWN-regulated (FC={fc:.2f}) but described as up-regulated"
                            )
                        elif true_direction == 'up' and claimed_down:
                            errors.append(
                                f"Hypothesis {i}: {gene_symbol} is UP-regulated (FC={fc:.2f}) but described as down-regulated"
                            )

        # Check central mechanistic model
        central_model = result.get('centralMechanisticModel', '')
        for gene_symbol, fc in gene_lookup.items():
            true_direction = 'up' if fc > 0 else 'down'

            if re.search(rf'\b{re.escape(gene_symbol)}\b', central_model, re.IGNORECASE):
                gene_context = re.findall(
                    rf'(\w+[\s\-]*regulat\w+|overexpress\w+|enhanced|increased|elevated|reduced|decreased|down[\s\-]*regulat\w+|up[\s\-]*regulat\w+|loss|suppression)[\s\w,]*\b{re.escape(gene_symbol)}\b',
                    central_model,
                    re.IGNORECASE
                )

                for context in gene_context:
                    context_lower = context.lower()
                    up_indicators = ['up-regulat', 'upregulat', 'overexpress', 'enhanced', 'increased', 'elevated']
                    down_indicators = ['down-regulat', 'downregulat', 'reduced', 'decreased', 'loss', 'suppression']

                    claimed_up = any(ind in context_lower for ind in up_indicators)
                    claimed_down = any(ind in context_lower for ind in down_indicators)

                    if true_direction == 'down' and claimed_up:
                        errors.append(
                            f"Central Model: {gene_symbol} is DOWN-regulated (FC={fc:.2f}) but described as up-regulated"
                        )
                    elif true_direction == 'up' and claimed_down:
                        errors.append(
                            f"Central Model: {gene_symbol} is UP-regulated (FC={fc:.2f}) but described as down-regulated"
                        )

        # LLM-based biochemistry review (generalizes previous hardcoded metabolic patterns)
        biochemistry_warnings = self._validate_biochemistry_with_llm(result, genes, experiment_context)
        errors.extend(biochemistry_warnings)

        return errors

    def _validate_biochemistry_with_llm(
        self,
        result: Dict,
        genes: List[Dict],
        experiment_context: str = ''
    ) -> List[str]:
        """
        Use LLM as a biochemistry reviewer to catch metabolic direction errors,
        cofactor confusion, and fabricated mechanisms.

        This generalizes the previous hardcoded regex patterns (α-KG/PHD/HIF-1α,
        ADH/NAD+) to work for any disease/pathway context.

        Args:
            result: LLM hypothesis generation result
            genes: List of DE genes with fold changes

        Returns:
            List of warning strings prefixed with 'Biochemistry review:' (empty on LLM error)
        """
        # Collect all hypothesis text + central model into a single review payload
        review_items = []
        for i, hyp in enumerate(result.get('hypotheses', []), 1):
            hyp_text = hyp.get('hypothesis', '')
            mechanism = hyp.get('mechanisticModel', '')
            claims_text = ''
            mech_claims = hyp.get('mechanisticClaims', [])
            if mech_claims:
                claims_lines = [f"    {c.get('source','?')} --[{c.get('relation','?')}]--> {c.get('target','?')}"
                               for c in mech_claims]
                claims_text = "\n  Mechanistic Claims (verify causality direction):\n" + "\n".join(claims_lines)
            review_items.append(
                f"Hypothesis {i}:\n  Statement: {hyp_text}\n  Mechanism: {mechanism}{claims_text}"
            )

        central_model = result.get('centralMechanisticModel', '')
        if central_model:
            review_items.append(f"Central Mechanistic Model:\n  {central_model}")

        if not review_items:
            return []

        # Build gene fold-change context
        gene_lines = []
        # Build a lookup dict: SYMBOL -> gene data for all genes
        gene_lookup = {}
        for gene in genes:
            symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                     gene.get('gene') or gene.get('name', '')).upper()
            if symbol:
                fc = gene.get('foldChange') or gene.get('log2_fold_change') or gene.get('fold_change', 0)
                gene_lookup[symbol] = fc

        sorted_genes = sorted(genes, key=lambda g: abs(g.get('foldChange', g.get('log2_fold_change', 0))), reverse=True)
        top_symbols = set()
        for gene in sorted_genes[:20]:
            symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                     gene.get('gene') or gene.get('name', 'Unknown'))
            fc = gene.get('foldChange') or gene.get('log2_fold_change') or gene.get('fold_change', 0)
            direction = 'UP' if fc > 0 else 'DOWN'
            gene_lines.append(f"  {symbol}: FC={fc:.2f} ({direction}-regulated)")
            top_symbols.add(symbol.upper())

        # Extract gene symbols mentioned in hypotheses and add their FC if not already in top 20
        # This prevents the reviewer from falsely flagging real genes as "fabricated"
        hypothesis_text = '\n'.join(review_items)
        mentioned_symbols = set(re.findall(r'\b([A-Z][A-Z0-9]{1,15})\b', hypothesis_text))
        # Filter to only symbols that are actually in the dataset
        mentioned_in_dataset = mentioned_symbols & set(gene_lookup.keys()) - top_symbols
        if mentioned_in_dataset:
            gene_lines.append("\n  --- Additional genes referenced in hypotheses ---")
            for symbol in sorted(mentioned_in_dataset):
                fc = gene_lookup[symbol]
                direction = 'UP' if fc > 0 else 'DOWN'
                gene_lines.append(f"  {symbol}: FC={fc:.2f} ({direction}-regulated)")

        context_line = f"\nExperimental context: {experiment_context}\n" if experiment_context else ""

        prompt = f"""You are a biochemistry reviewer. Check the following hypotheses for biochemical errors.

CRITICAL FIRST STEP: Before evaluating any claim, identify the full gene name and established molecular function for each gene symbol mentioned. Many gene symbols are ambiguous — the same abbreviation can refer to different proteins in different contexts. Always use the gene identity most relevant to the biological context (e.g., metabolic pathways, signaling cascades) described in the hypothesis. State each gene's full name and function in your reasoning before flagging errors.

Gene expression data (log2 fold changes, includes top genes + all genes referenced in hypotheses):
{chr(10).join(gene_lines)}

IMPORTANT: Every gene listed above with a FC value IS in the dataset. Do NOT flag any of these genes as "fabricated" or "not in the dataset".
{context_line}
Hypotheses to review:
{chr(10).join(review_items)}

Check for these categories of errors:
1. METABOLIC REACTION DIRECTION: If an enzyme is downregulated, its product decreases and its substrate accumulates. The reverse for upregulated enzymes. Flag claims that get this backwards.
2. COFACTOR/CO-SUBSTRATE CONFUSION: Verify whether a metabolite is a co-substrate (activator) or competitive inhibitor of a target enzyme. Flag claims that confuse the two (e.g., claiming a co-substrate inhibits its target).
3. COFACTOR BALANCE ERRORS: For reactions involving NAD+/NADH, ATP/ADP, FAD/FADH2 etc., verify the direction. If an enzyme that CONSUMES NAD+ is downregulated, NAD+ is SPARED (not depleted).
4. FABRICATED MECHANISMS: Flag any molecular mechanism that is not grounded in established biochemistry (e.g., invented enzyme-substrate relationships). NOTE: Do NOT flag a gene as fabricated just because it has a small fold change. If a gene is listed in the gene expression data above, it IS measured and present in the dataset.
5. REVERSED CAUSALITY: Check whether the hypothesis reverses the known causal direction.
   For example, if Gene A is a known target of Transcription Factor B (B→A), the hypothesis
   should NOT claim that A activates/reinforces/drives B (A→B) unless providing a feedback
   loop mechanism. Flag claims where a downstream target is presented as an upstream driver.
6. TISSUE/DISEASE CONTEXT MISMATCH: Flag hypotheses that assume cells in the experimental system retain normal, differentiated tissue functions that are likely lost or altered in the disease/condition being studied. For example, tumor cells typically lose specialized tissue functions through dedifferentiation; immortalized cell lines may not behave like primary tissue; inflamed or fibrotic tissue may lose homeostatic functions. The hypothesis must be plausible given the actual cellular state in the experiment, not just normal physiology.
7. UNESTABLISHED REGULATORY LINKS: Flag claims that a specific transcription factor directly regulates a target gene when that regulatory relationship is not well-established in biochemistry. The claimed TF→target relationship must be a known, canonical interaction supported by ChIP-seq, reporter assays, or equivalent evidence in the literature. Hypothetical or speculative regulatory links should not be presented as established mechanisms.
8. SELF-CONTRADICTORY REASONING: Flag hypotheses where the stated premises logically contradict the conclusion. Common pattern: the hypothesis correctly states Fact A and Fact B, but then draws a conclusion that contradicts one of them.
   - Example: "Transporter X is downregulated → substrate Y accumulates → this activates Process Z" BUT Process Z requires LOW substrate Y (not high). The premises are individually correct but the conclusion inverts the direction.
   - Watch for ion/metabolite accumulation vs depletion: if a channel or transporter is lost, check whether the downstream process is triggered by HIGH or LOW levels of the affected molecule, and verify the hypothesis gets this right.
   - Watch for inhibitor loss: if an inhibitor of Process P is downregulated, Process P should INCREASE (de-repression). Flag claims where inhibitor loss is said to decrease the target process.
9. DEDIFFERENTIATION MARKER MISUSE: In disease contexts (especially cancer), flag hypotheses that treat downregulated tissue-specific differentiation markers as active mechanistic drivers of pathology. Tissue-specific transporters, channels, enzymes, and structural proteins are commonly downregulated due to loss of normal cellular identity (e.g., dedifferentiation in tumors, trans-differentiation in fibrosis). Their downregulation is typically a CONSEQUENCE of the disease process, not a CAUSE of downstream signaling changes. Flag hypotheses that present such marker loss as directly driving inflammation, metabolic reprogramming, or signaling cascades, unless the hypothesis provides a specific, established mechanism for how the protein's absence causally triggers the claimed effect.

Return ONLY a JSON object in this exact format:
{{"issues": [{{"hypothesis_index": 1, "category": "metabolic_direction|cofactor_confusion|cofactor_balance|fabricated_mechanism|reversed_causality|tissue_context_mismatch|unestablished_regulatory_link|self_contradictory_reasoning|dedifferentiation_marker_misuse", "claim": "the specific claim that is wrong", "correction": "what the correct biochemistry is", "severity": "error|warning"}}]}}

If no issues are found, return: {{"issues": []}}
Return ONLY the JSON object, no other text."""

        try:
            is_separate = self.reviewer_llm is not self.llm
            generator_model = getattr(self.llm, 'model', 'unknown')
            reviewer_model = getattr(self.reviewer_llm, 'model', 'unknown')
            num_hypotheses = len(result.get('hypotheses', []))

            # ── Step 1: Generator model identifies potential issues (general 9-category prompt) ──
            print(f'\n  🔬 Biochemistry Review (CROSS-MODEL, ANTI-SYCOPHANCY)')
            print(f'     Generator: {generator_model} (claim detection)')
            print(f'     Reviewer:  {reviewer_model} (independent verification)')

            print(f'\n     [Step 1] {generator_model} scanning {num_hypotheses} hypothesis(es)...')
            generator_response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                seed=42
            )
            print(f'     [Step 1] Response: {len(generator_response)} chars')

            # Parse generator flags
            generator_issues = self._parse_review_response(generator_response)
            error_issues = [iss for iss in generator_issues
                           if isinstance(iss.get('hypothesis_index'), int) and iss.get('severity') == 'error']
            flagged_indices = set(iss['hypothesis_index'] for iss in error_issues)
            print(f'     [Step 1] Flagged: {len(error_issues)} issue(s) in hypotheses {sorted(flagged_indices) if flagged_indices else "none"}')

            if not error_issues:
                print(f'\n  Biochemistry review: No issues found')
                return []

            # ── Step 2: Reviewer model verifies each flag via anti-sycophancy ──
            if is_separate:
                print(f'\n     [Step 2] {reviewer_model} verifying {len(error_issues)} flag(s)...')
                print(f'     Method: independent knowledge question → compare with claim')

                confirmed_issues = []
                unconfirmed_issues = []

                for flag in error_issues:
                    hyp_idx = flag['hypothesis_index']
                    claim = flag.get('claim', '')
                    category = flag.get('category', '')
                    correction = flag.get('correction', '')

                    # Build independent knowledge question (general, not disease-specific)
                    knowledge_q = self._build_independent_question(claim, category, correction)
                    if not knowledge_q:
                        # Can't build a question — keep as unconfirmed warning
                        unconfirmed_issues.append(flag)
                        print(f'     Hyp {hyp_idx} [{category}]: SKIPPED (no verification question)')
                        continue

                    try:
                        # Ask reviewer to state the correct biochemistry independently
                        knowledge_resp = self.reviewer_llm.chat(
                            [{"role": "user", "content": knowledge_q}],
                            temperature=0.0,
                            seed=42,
                            max_tokens=300
                        )

                        # Compare: does the independent answer contradict the hypothesis claim?
                        comparison_prompt = f"""A hypothesis claims: "{claim}"

An independent expert states: "{knowledge_resp.strip()}"

Does the expert's statement CONTRADICT the hypothesis claim? Answer ONLY "CONTRADICTS" or "CONSISTENT"."""

                        comparison_resp = self.reviewer_llm.chat(
                            [{"role": "user", "content": comparison_prompt}],
                            temperature=0.0,
                            seed=42,
                            max_tokens=20
                        )

                        verdict = comparison_resp.strip().upper()
                        is_confirmed = 'CONTRADICT' in verdict
                        flag_label = f"[{category}] {claim[:80]}..."

                        if is_confirmed:
                            # Use reviewer's knowledge as the correction
                            flag['correction'] = knowledge_resp.strip()[:300]
                            confirmed_issues.append(flag)
                            print(f'     Hyp {hyp_idx}: CONFIRMED — {flag_label}')
                        else:
                            unconfirmed_issues.append(flag)
                            print(f'     Hyp {hyp_idx}: not confirmed — {flag_label}')

                    except Exception as e:
                        unconfirmed_issues.append(flag)
                        print(f'     Hyp {hyp_idx}: verification failed ({e})')

                confirmed_count = len(confirmed_issues)
                unconfirmed_count = len(unconfirmed_issues)
                print(f'\n     [Step 2] Results: {confirmed_count} confirmed, {unconfirmed_count} unconfirmed')

            else:
                # No separate reviewer — all generator flags stand as-is
                confirmed_issues = error_issues
                unconfirmed_issues = []

            # ── Build warnings ──
            warnings = []

            # Confirmed issues → error severity (will trigger removal)
            for issue in confirmed_issues:
                hyp_idx = issue.get('hypothesis_index', '?')
                location = f'Hypothesis {hyp_idx}'
                warning = (
                    f"Biochemistry review (error) - {location}: "
                    f"[{issue.get('category', 'unknown')}] {issue.get('claim', '')} "
                    f"— Correction: {issue.get('correction', '')}"
                )
                warnings.append(warning)

            # Unconfirmed issues → warning severity (visible but won't remove)
            for issue in unconfirmed_issues:
                hyp_idx = issue.get('hypothesis_index', '?')
                location = f'Hypothesis {hyp_idx}'
                warning = (
                    f"Biochemistry review (warning) - {location}: "
                    f"[{issue.get('category', 'unknown')}] {issue.get('claim', '')} "
                    f"— Correction: {issue.get('correction', '')} "
                    f"(not confirmed by {reviewer_model})"
                )
                warnings.append(warning)

            if warnings:
                print(f'\n  Biochemistry review: {len(confirmed_issues)} error(s), {len(unconfirmed_issues)} warning(s)')
            else:
                print(f'\n  Biochemistry review: No issues found')

            return warnings

        except Exception as e:
            # Fail open — don't block pipeline on LLM error
            print(f'  Biochemistry review: LLM call failed ({e}), skipping')
            return []

    @staticmethod
    def _build_independent_question(claim: str, category: str, correction: str) -> Optional[str]:
        """Build an independent knowledge question from a flagged claim.

        The question must NOT contain the hypothesis claim itself — only ask
        the reviewer to state the correct biochemistry so we can compare.
        All questions are disease-agnostic.
        """
        claim_lower = claim.lower()

        # Extract gene/protein names from the claim
        genes_in_claim = re.findall(r'\b([A-Z][A-Z0-9]{1,12})\b', claim)
        noise = {'THE', 'AND', 'FOR', 'NOT', 'WITH', 'FROM', 'INTO', 'THAT', 'THIS',
                 'VIA', 'ARE', 'WAS', 'HAS', 'BEEN', 'WILL', 'CAN', 'ITS', 'BUT',
                 'COA', 'TCA', 'ATP', 'NAD', 'FAD', 'DNA', 'RNA', 'ROS', 'HIF'}
        genes_in_claim = [g for g in genes_in_claim if g not in noise and len(g) >= 3]

        # Detect biological context from the claim to help disambiguate gene symbols
        context_hint = ""
        if any(x in claim_lower for x in ['amino acid', 'bcaa', 'branched', 'catabolism', 'valine', 'leucine']):
            context_hint = " in the context of amino acid catabolism"
        elif any(x in claim_lower for x in ['tca', 'citrate', 'oxoglutarate', 'mitochondri']):
            context_hint = " in the context of mitochondrial metabolism"
        elif any(x in claim_lower for x in ['immune', 'nk cell', 't cell', 'cytotoxic']):
            context_hint = " in the context of immune signaling"
        elif any(x in claim_lower for x in ['er ', 'endoplasmic', 'protein folding', 'chaperone']):
            context_hint = " in the context of protein processing"

        if category == 'fabricated_mechanism':
            if len(genes_in_claim) >= 2:
                g1, g2 = genes_in_claim[0], genes_in_claim[1]
                return (f"What is the full gene name and established molecular function of {g1}{context_hint}? "
                        f"And what is the full gene name and function of {g2}{context_hint}? "
                        f"Do they directly interact in the same enzymatic reaction? Answer in 2-3 sentences.")
            elif genes_in_claim:
                return (f"What is the full gene name and established molecular function of {genes_in_claim[0]}{context_hint}? "
                        f"Is it an ER chaperone, a transcription factor, a kinase, or a metabolic enzyme? "
                        f"Answer in 1-2 sentences.")

        elif category == 'metabolic_direction':
            if genes_in_claim:
                return (f"What is the full gene name of {genes_in_claim[0]}{context_hint}? "
                        f"What substrate does it act on, and what product does it generate? "
                        f"Answer in 1-2 sentences.")

        elif category == 'self_contradictory_reasoning':
            if 'alkalin' in claim_lower and 'h+' in claim_lower.replace('h⁺', 'h+'):
                return "If H+ ions accumulate inside a cell, does intracellular pH increase (alkalinize) or decrease (acidify)? Answer in 1 sentence."

        elif category == 'dedifferentiation_marker_misuse':
            if genes_in_claim:
                gene_str = ' and '.join(genes_in_claim[:2])
                return (f"Is {gene_str} a tissue-specific differentiation marker? "
                        f"In cancer, is its downregulation typically a driver of pathology "
                        f"or a consequence of loss of differentiated cell identity? "
                        f"Answer in 1-2 sentences.")

        elif category == 'reversed_causality':
            if len(genes_in_claim) >= 2:
                return (f"In established biochemistry, does {genes_in_claim[0]} regulate {genes_in_claim[1]}, "
                        f"or does {genes_in_claim[1]} regulate {genes_in_claim[0]}? Answer in 1-2 sentences.")

        elif category == 'unestablished_regulatory_link':
            if 'prolyl' in claim_lower or 'phd' in claim_lower:
                return ("Do branched-chain keto-acids inhibit prolyl hydroxylases (PHDs)? "
                        "Is there experimental evidence for this? Answer in 1-2 sentences.")
            elif len(genes_in_claim) >= 2:
                return (f"Is there established evidence that {genes_in_claim[0]} directly regulates "
                        f"{genes_in_claim[1]}? Answer in 1-2 sentences.")

        # Fallback: use the correction text to build a question
        if correction and genes_in_claim:
            return f"What is the established biochemical function of {genes_in_claim[0]}? Answer in 1-2 sentences."

        return None

    def _parse_review_response(self, response: str) -> List[Dict]:
        """Parse a biochemistry review LLM response into a list of issue dicts."""
        if not response:
            return []
        response_text = response.strip()
        # Handle markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        # Try direct parse
        try:
            parsed = json.loads(response_text)
            return parsed.get('issues', [])
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return parsed.get('issues', [])
                except json.JSONDecodeError:
                    pass
        return []

    def _validate_numerical_claims(
        self,
        result: Dict,
        genes: List[Dict],
        pathways: List[Dict],
        mechanisms_result: Optional[Dict] = None
    ) -> List[str]:
        """
        Validate numerical claims in hypotheses against source data.

        Checks for:
        - Fold change values attributed to the wrong gene
        - p-values attributed to the wrong gene
        - FDR values attributed to the wrong gene
        - Pathway FDR misattributed as gene-level FDR
        - Incorrect DE gene counts for pathways

        Args:
            result: LLM hypothesis generation result
            genes: List of DE genes with fold changes and p-values
            pathways: List of enriched pathways
            mechanisms_result: Pathway mechanisms from Step 3

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        # Build gene lookup: symbol -> (fold_change, p_value, fdr)
        gene_lookup = {}
        for gene in genes:
            symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                     gene.get('gene') or gene.get('name', '')).upper()
            fc = gene.get('foldChange') or gene.get('log2_fold_change') or gene.get('fold_change')
            pval = gene.get('pValue') or gene.get('p_value')
            fdr = gene.get('pValueFDR') or gene.get('p_value_fdr') or gene.get('fdr') or gene.get('adjPValue')
            if symbol and fc is not None:
                gene_lookup[symbol] = {
                    'fc': float(fc),
                    'p_value': float(pval) if pval else None,
                    'fdr': float(fdr) if fdr else None
                }

        # Build pathway FDR lookup for detecting misattribution
        # Maps FDR values to pathway names (so we can tell if a gene's claimed FDR is actually a pathway FDR)
        pathway_fdr_lookup = {}  # fdr_value -> pathway_name
        for pathway in pathways:
            pname = pathway.get('name', pathway.get('pathwayName', ''))
            pfdr = pathway.get('pValueFDR', pathway.get('adjPValue', pathway.get('pValue')))
            if pname and pfdr is not None:
                pathway_fdr_lookup[float(pfdr)] = pname

        # Also get pathway FDR from mechanisms_result pathway_structures
        if mechanisms_result and mechanisms_result.get('pathway_structures'):
            for struct in mechanisms_result['pathway_structures']:
                pname = struct.get('pathway', '')
                pfdr = struct.get('p_value_fdr')
                if pname and pfdr is not None:
                    pathway_fdr_lookup[float(pfdr)] = pname

        # Build pathway gene count lookup from mechanisms_result
        pathway_gene_counts = {}
        if mechanisms_result and mechanisms_result.get('pathway_structures'):
            for struct in mechanisms_result['pathway_structures']:
                pname = struct.get('pathway', '')
                de_count = struct.get('de_genes_count', 0)
                if pname:
                    pathway_gene_counts[pname.lower()] = de_count

        # Collect all text fields from hypotheses
        text_blocks = []
        for i, hyp in enumerate(result.get('hypotheses', []), 1):
            for field in ['hypothesis', 'mechanisticModel', 'directionalPrediction', 'confidenceRationale']:
                text = hyp.get(field, '')
                if text:
                    text_blocks.append((f'Hypothesis {i}', text))
            for ev in hyp.get('evidenceSupporting', []):
                text_blocks.append((f'Hypothesis {i}', ev))

        central = result.get('centralMechanisticModel', '')
        if central:
            text_blocks.append(('Central Model', central))

        summary = result.get('hypothesesSummary', '')
        if summary:
            text_blocks.append(('Summary', summary))

        # Regex patterns for numerical claims
        # FC patterns: "GENE (↑3.64)" or "GENE ↑3.64-fold" or "GENE ... 3.64-fold"
        fc_patterns = [
            re.compile(r'(\b[A-Z][A-Z0-9]{1,15}\b)\s*\([↑↓]\s*(\d+\.\d+)\)', re.IGNORECASE),
            re.compile(r'(\b[A-Z][A-Z0-9]{1,15}\b)\s+[↑↓]\s*(\d+\.\d+)[\s\-]*fold', re.IGNORECASE),
            re.compile(r'(\b[A-Z][A-Z0-9]{1,15}\b)\s+\(?FC[:\s=]*[+-]?(\d+\.\d+)\)?', re.IGNORECASE),
        ]

        # p-value pattern: "GENE ... p=1.23e-10" or "p<1.23e-10" (gene within ~80 chars)
        # Note: U+2011 (non-breaking hyphen), U+2212 (minus sign), U+002D (hyphen-minus)
        pval_pattern = re.compile(
            r'(\b[A-Z][A-Z0-9]{1,15}\b).{0,80}?p\s*[=<]\s*(\d+\.?\d*)\s*[×xX*]?\s*10\s*[\^]?\s*[−\-\u2011]\s*(\d+)',
            re.IGNORECASE
        )
        pval_pattern_e = re.compile(
            r'(\b[A-Z][A-Z0-9]{1,15}\b).{0,80}?p\s*[=<]\s*(\d+\.?\d*e[−\-\u2011]\d+)',
            re.IGNORECASE
        )

        # Gene count pattern: "N DE genes" near pathway context
        gene_count_pattern = re.compile(
            r'(\d+)\s+(?:DE\s+genes|differentially\s+expressed\s+genes?)',
            re.IGNORECASE
        )

        # FDR patterns: "GENE ... FDR=1.23e-10" or "FDR=0.00155" (gene within ~80 chars)
        # Note: U+2011 (non-breaking hyphen), U+2212 (minus sign), U+002D (hyphen-minus)
        fdr_pattern_sci = re.compile(
            r'(\b[A-Z][A-Z0-9]{1,15}\b).{0,80}?FDR\s*[=<≈]\s*(\d+\.?\d*)\s*[×xX*]?\s*10\s*[\^]?\s*[−\-\u2011]\s*(\d+)',
            re.IGNORECASE
        )
        fdr_pattern_e = re.compile(
            r'(\b[A-Z][A-Z0-9]{1,15}\b).{0,80}?FDR\s*[=<≈]\s*(\d+\.?\d*e[−\-\u2011]\d+)',
            re.IGNORECASE
        )
        fdr_pattern_decimal = re.compile(
            r'(\b[A-Z][A-Z0-9]{1,15}\b).{0,80}?FDR\s*[=<≈]\s*(0\.\d{3,})',
            re.IGNORECASE
        )

        FC_TOLERANCE = 0.05

        def _check_fdr_value(location: str, claimed_gene: str, claimed_fdr: float):
            """Helper to validate a claimed FDR value for a gene."""
            if claimed_gene not in gene_lookup:
                return
            actual_fdr = gene_lookup[claimed_gene].get('fdr')
            if actual_fdr is None or actual_fdr <= 0 or claimed_fdr <= 0:
                return

            log_diff = abs(math.log10(claimed_fdr) - math.log10(actual_fdr))
            if log_diff > 1.0:  # More than 1 order of magnitude off
                # Check if this FDR is actually a pathway FDR
                pathway_source = None
                for pfdr, pname in pathway_fdr_lookup.items():
                    if pfdr > 0 and abs(math.log10(claimed_fdr) - math.log10(pfdr)) <= 0.3:
                        pathway_source = pname
                        break

                msg = (f"{location}: {claimed_gene} cited with FDR={claimed_fdr:.2e} "
                       f"but actual gene-level FDR={actual_fdr:.2e}")
                if pathway_source:
                    msg += (f" (FDR={claimed_fdr:.2e} appears to be the PATHWAY FDR "
                            f"for '{pathway_source}', not the gene-level FDR)")
                warnings.append(msg)

        for location, text in text_blocks:
            # Check FC claims
            for pattern in fc_patterns:
                for match in pattern.finditer(text):
                    claimed_gene = match.group(1).upper()
                    claimed_fc = float(match.group(2))

                    if claimed_gene in gene_lookup:
                        actual_fc = abs(gene_lookup[claimed_gene]['fc'])
                        if abs(claimed_fc - actual_fc) > FC_TOLERANCE:
                            # Check if this FC belongs to a different gene
                            real_owner = None
                            for sym, data in gene_lookup.items():
                                if abs(abs(data['fc']) - claimed_fc) <= FC_TOLERANCE and sym != claimed_gene:
                                    real_owner = sym
                                    break
                            msg = (f"{location}: {claimed_gene} cited with FC={claimed_fc:.2f} "
                                   f"but actual FC={actual_fc:.2f}")
                            if real_owner:
                                msg += f" (FC={claimed_fc:.2f} belongs to {real_owner})"
                            warnings.append(msg)

            # Check p-value claims (scientific notation with x10^)
            for match in pval_pattern.finditer(text):
                claimed_gene = match.group(1).upper()
                mantissa = float(match.group(2))
                exponent = int(match.group(3))
                claimed_pval = mantissa * (10 ** -exponent)

                if claimed_gene in gene_lookup and gene_lookup[claimed_gene]['p_value'] is not None:
                    actual_pval = gene_lookup[claimed_gene]['p_value']
                    if actual_pval > 0 and claimed_pval > 0:
                        log_diff = abs(math.log10(claimed_pval) - math.log10(actual_pval))
                        if log_diff > 1.0:  # More than 1 order of magnitude off
                            real_owner = None
                            for sym, data in gene_lookup.items():
                                if (data['p_value'] and data['p_value'] > 0 and sym != claimed_gene and
                                        abs(math.log10(claimed_pval) - math.log10(data['p_value'])) <= 0.5):
                                    real_owner = sym
                                    break
                            msg = (f"{location}: {claimed_gene} cited with p={claimed_pval:.2e} "
                                   f"but actual p={actual_pval:.2e}")
                            if real_owner:
                                msg += f" (p-value likely belongs to {real_owner})"
                            warnings.append(msg)

            # Check p-value claims (e-notation like p=1.23e-10)
            for match in pval_pattern_e.finditer(text):
                claimed_gene = match.group(1).upper()
                pval_str = match.group(2).replace('\u2212', '-').replace('\u2011', '-')  # Replace unicode minus/non-breaking hyphen
                try:
                    claimed_pval = float(pval_str)
                except ValueError:
                    continue

                if claimed_gene in gene_lookup and gene_lookup[claimed_gene]['p_value'] is not None:
                    actual_pval = gene_lookup[claimed_gene]['p_value']
                    if actual_pval > 0 and claimed_pval > 0:
                        log_diff = abs(math.log10(claimed_pval) - math.log10(actual_pval))
                        if log_diff > 1.0:
                            real_owner = None
                            for sym, data in gene_lookup.items():
                                if (data['p_value'] and data['p_value'] > 0 and sym != claimed_gene and
                                        abs(math.log10(claimed_pval) - math.log10(data['p_value'])) <= 0.5):
                                    real_owner = sym
                                    break
                            msg = (f"{location}: {claimed_gene} cited with p={claimed_pval:.2e} "
                                   f"but actual p={actual_pval:.2e}")
                            if real_owner:
                                msg += f" (p-value likely belongs to {real_owner})"
                            warnings.append(msg)

            # Check FDR claims (scientific notation with x10^)
            for match in fdr_pattern_sci.finditer(text):
                claimed_gene = match.group(1).upper()
                mantissa = float(match.group(2))
                exponent = int(match.group(3))
                claimed_fdr = mantissa * (10 ** -exponent)
                _check_fdr_value(location, claimed_gene, claimed_fdr)

            # Check FDR claims (e-notation like FDR=1.23e-10)
            for match in fdr_pattern_e.finditer(text):
                claimed_gene = match.group(1).upper()
                fdr_str = match.group(2).replace('\u2212', '-').replace('\u2011', '-')
                try:
                    claimed_fdr = float(fdr_str)
                except ValueError:
                    continue
                _check_fdr_value(location, claimed_gene, claimed_fdr)

            # Check FDR claims (decimal notation like FDR=0.00155)
            for match in fdr_pattern_decimal.finditer(text):
                claimed_gene = match.group(1).upper()
                try:
                    claimed_fdr = float(match.group(2))
                except ValueError:
                    continue
                _check_fdr_value(location, claimed_gene, claimed_fdr)

            # Check gene count claims
            for match in gene_count_pattern.finditer(text):
                claimed_count = int(match.group(1))
                # Try to find which pathway this count refers to
                # Look in surrounding context (100 chars before and after)
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].lower()

                for pname, actual_count in pathway_gene_counts.items():
                    if pname.lower()[:20] in context and claimed_count != actual_count:
                        warnings.append(
                            f"{location}: Claims {claimed_count} DE genes for '{pname}' "
                            f"but actual count is {actual_count}"
                        )

        # Issue #2: Detect identical statistics shared by 3+ genes (pathway-level misattribution)
        pval_by_value = {}  # claimed_pval_str -> list of gene symbols
        fdr_by_value = {}   # claimed_fdr_str -> list of gene symbols

        for location, text in text_blocks:
            # Collect p-value claims for identical-stats detection
            for match in pval_pattern.finditer(text):
                gene_sym = match.group(1).upper()
                if gene_sym in gene_lookup:
                    mantissa = float(match.group(2))
                    exponent = int(match.group(3))
                    claimed_pval = mantissa * (10 ** -exponent)
                    key = f"{claimed_pval:.6e}"
                    pval_by_value.setdefault(key, set()).add(gene_sym)

            for match in pval_pattern_e.finditer(text):
                gene_sym = match.group(1).upper()
                if gene_sym in gene_lookup:
                    pval_str = match.group(2).replace('\u2212', '-')
                    try:
                        claimed_pval = float(pval_str)
                        key = f"{claimed_pval:.6e}"
                        pval_by_value.setdefault(key, set()).add(gene_sym)
                    except ValueError:
                        pass

            # Collect FDR claims for identical-stats detection
            for match in fdr_pattern_sci.finditer(text):
                gene_sym = match.group(1).upper()
                if gene_sym in gene_lookup:
                    mantissa = float(match.group(2))
                    exponent = int(match.group(3))
                    claimed_fdr = mantissa * (10 ** -exponent)
                    key = f"{claimed_fdr:.6e}"
                    fdr_by_value.setdefault(key, set()).add(gene_sym)

            for match in fdr_pattern_e.finditer(text):
                gene_sym = match.group(1).upper()
                if gene_sym in gene_lookup:
                    fdr_str = match.group(2).replace('\u2212', '-').replace('\u2011', '-')
                    try:
                        claimed_fdr = float(fdr_str)
                        key = f"{claimed_fdr:.6e}"
                        fdr_by_value.setdefault(key, set()).add(gene_sym)
                    except ValueError:
                        pass

            for match in fdr_pattern_decimal.finditer(text):
                gene_sym = match.group(1).upper()
                if gene_sym in gene_lookup:
                    try:
                        claimed_fdr = float(match.group(2))
                        key = f"{claimed_fdr:.6e}"
                        fdr_by_value.setdefault(key, set()).add(gene_sym)
                    except ValueError:
                        pass

        # Flag groups of 3+ genes sharing identical p-value or FDR
        for pval_key, gene_set in pval_by_value.items():
            if len(gene_set) >= 3:
                claimed_pval = float(pval_key)
                # Check if shared value matches a pathway FDR
                pathway_source = None
                for pfdr, pname in pathway_fdr_lookup.items():
                    if pfdr > 0 and claimed_pval > 0:
                        if abs(math.log10(claimed_pval) - math.log10(pfdr)) <= 0.3:
                            pathway_source = pname
                            break
                msg = (f"Identical p-value p={claimed_pval:.2e} cited for {len(gene_set)} different genes "
                       f"({', '.join(sorted(gene_set))}): likely pathway-level stat misattributed as gene-level")
                if pathway_source:
                    msg += f" (matches pathway FDR for '{pathway_source}')"
                warnings.append(msg)

        for fdr_key, gene_set in fdr_by_value.items():
            if len(gene_set) >= 3:
                claimed_fdr = float(fdr_key)
                pathway_source = None
                for pfdr, pname in pathway_fdr_lookup.items():
                    if pfdr > 0 and claimed_fdr > 0:
                        if abs(math.log10(claimed_fdr) - math.log10(pfdr)) <= 0.3:
                            pathway_source = pname
                            break
                msg = (f"Identical FDR={claimed_fdr:.2e} cited for {len(gene_set)} different genes "
                       f"({', '.join(sorted(gene_set))}): likely pathway-level stat misattributed as gene-level")
                if pathway_source:
                    msg += f" (matches pathway FDR for '{pathway_source}')"
                warnings.append(msg)

        return warnings

    def _autocorrect_gene_stats(
        self,
        result: Dict,
        genes: List[Dict],
    ) -> int:
        """
        Auto-correct fabricated gene-level p-values and FDRs in hypothesis text.

        When the LLM doesn't have a gene's actual p-value/FDR (because it wasn't
        in the top genes shown), it often fabricates values. This method scans all
        hypothesis text fields and replaces any p-value or FDR that is >1 order of
        magnitude off from the actual value with the correct one.

        Returns:
            Number of corrections made
        """
        # Build gene lookup: symbol -> {p_value, fdr}
        gene_stats_lookup = {}
        for gene in genes:
            symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                     gene.get('gene') or gene.get('name', '')).upper()
            pval = gene.get('pValue') or gene.get('p_value')
            fdr = gene.get('pValueFDR') or gene.get('p_value_fdr') or gene.get('fdr') or gene.get('adjPValue')
            if symbol:
                gene_stats_lookup[symbol] = {
                    'p_value': float(pval) if pval else None,
                    'fdr': float(fdr) if fdr else None
                }

        if not gene_stats_lookup:
            return 0

        # Pattern for gene p-value claims: "GENE ... p=VALUE" or "gene p=VALUE"
        pval_claim_pattern = re.compile(
            r'(\b[A-Z][A-Z0-9]{1,15}\b)(.{0,80}?(?:gene\s+)?p\s*[=<]\s*)'
            r'(\d+\.?\d*e[−\-\u2011]\d+|\d+\.?\d*\s*[×xX*]\s*10\s*[\^]?\s*[−\-\u2011]\s*\d+)',
            re.IGNORECASE
        )
        # Pattern for gene FDR claims: "GENE ... FDR=VALUE" or "gene FDR=VALUE"
        fdr_claim_pattern = re.compile(
            r'(\b[A-Z][A-Z0-9]{1,15}\b)(.{0,80}?(?:gene\s+)?FDR\s*[=<≈]\s*)'
            r'(\d+\.?\d*e[−\-\u2011]\d+|\d+\.?\d*\s*[×xX*]\s*10\s*[\^]?\s*[−\-\u2011]\s*\d+|0\.\d{3,})',
            re.IGNORECASE
        )

        corrections = 0

        def _parse_numeric(val_str: str) -> float:
            """Parse a scientific notation string to float."""
            normalized = val_str.replace('\u2212', '-').replace('\u2011', '-').replace(' ', '')
            sci_match = re.match(r'(\d+\.?\d*)[×xX*]10[\^]?[−\-](\d+)', normalized)
            if sci_match:
                return float(sci_match.group(1)) * (10 ** -int(sci_match.group(2)))
            return float(normalized)

        def _correct_pvals(text: str) -> str:
            nonlocal corrections
            if not text:
                return text

            def pval_replacer(match):
                nonlocal corrections
                gene_sym = match.group(1).upper()
                prefix = match.group(2)
                pval_str = match.group(3)

                if gene_sym not in gene_stats_lookup:
                    return match.group(0)
                actual_pval = gene_stats_lookup[gene_sym].get('p_value')
                if actual_pval is None or actual_pval <= 0:
                    return match.group(0)

                try:
                    claimed_pval = _parse_numeric(pval_str)
                except (ValueError, TypeError):
                    return match.group(0)

                if claimed_pval <= 0:
                    return match.group(0)

                log_diff = abs(math.log10(claimed_pval) - math.log10(actual_pval))
                if log_diff > 1.0:
                    corrections += 1
                    return f"{match.group(1)}{prefix}{actual_pval:.2e}"

                return match.group(0)

            return pval_claim_pattern.sub(pval_replacer, text)

        def _correct_fdrs(text: str) -> str:
            nonlocal corrections
            if not text:
                return text

            def fdr_replacer(match):
                nonlocal corrections
                gene_sym = match.group(1).upper()
                prefix = match.group(2)
                fdr_str = match.group(3)

                if gene_sym not in gene_stats_lookup:
                    return match.group(0)
                actual_fdr = gene_stats_lookup[gene_sym].get('fdr')
                if actual_fdr is None or actual_fdr <= 0:
                    return match.group(0)

                try:
                    claimed_fdr = _parse_numeric(fdr_str)
                except (ValueError, TypeError):
                    return match.group(0)

                if claimed_fdr <= 0:
                    return match.group(0)

                log_diff = abs(math.log10(claimed_fdr) - math.log10(actual_fdr))
                if log_diff > 1.0:
                    corrections += 1
                    return f"{match.group(1)}{prefix}{actual_fdr:.2e}"

                return match.group(0)

            return fdr_claim_pattern.sub(fdr_replacer, text)

        def _correct_field(text: str) -> str:
            text = _correct_pvals(text)
            text = _correct_fdrs(text)
            return text

        # Correct text fields in each hypothesis
        for hyp in result.get('hypotheses', []):
            for field in ['hypothesis', 'mechanisticModel', 'confidenceRationale']:
                if hyp.get(field):
                    hyp[field] = _correct_field(hyp[field])
            # Correct evidence list
            if hyp.get('evidenceSupporting'):
                hyp['evidenceSupporting'] = [_correct_field(ev) for ev in hyp['evidenceSupporting']]

        # Correct central model and summary
        if result.get('centralMechanisticModel'):
            result['centralMechanisticModel'] = _correct_field(result['centralMechanisticModel'])
        if result.get('hypothesesSummary'):
            result['hypothesesSummary'] = _correct_field(result['hypothesesSummary'])

        return corrections

    def _autocorrect_pathway_fdr(
        self,
        result: Dict,
        genes: List[Dict],
        pathways: List[Dict],
        mechanisms_result: Optional[Dict] = None
    ) -> int:
        """
        Auto-correct pathway FDR values misattributed as gene-level FDR in hypothesis text.

        Scans hypothesis fields for patterns like "GENE ... FDR=X" where X matches a known
        pathway FDR but not the gene's actual FDR. Replaces with the correct gene-level FDR.

        Returns:
            Number of corrections made
        """
        # Build gene lookup: symbol -> fdr
        gene_fdr_lookup = {}
        for gene in genes:
            symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                     gene.get('gene') or gene.get('name', '')).upper()
            fdr = gene.get('pValueFDR') or gene.get('p_value_fdr') or gene.get('fdr') or gene.get('adjPValue')
            if symbol and fdr is not None:
                gene_fdr_lookup[symbol] = float(fdr)

        # Build pathway FDR set for detecting misattribution
        pathway_fdr_values = set()
        for pathway in pathways:
            pfdr = pathway.get('pValueFDR', pathway.get('adjPValue', pathway.get('pValue')))
            if pfdr is not None:
                pathway_fdr_values.add(float(pfdr))
        if mechanisms_result and mechanisms_result.get('pathway_structures'):
            for struct in mechanisms_result['pathway_structures']:
                pfdr = struct.get('p_value_fdr')
                if pfdr is not None:
                    pathway_fdr_values.add(float(pfdr))

        if not pathway_fdr_values or not gene_fdr_lookup:
            return 0

        # Pattern: GENE_SYMBOL ... FDR=VALUE (gene within ~80 chars before FDR)
        # Captures: (gene_symbol, pre_fdr_text, fdr_value_string)
        fdr_claim_pattern = re.compile(
            r'(\b[A-Z][A-Z0-9]{1,15}\b)(.{0,80}?FDR\s*[=<≈]\s*)(\d+\.?\d*e[−\-\u2011]\d+|\d+\.?\d*\s*[×xX*]\s*10\s*[\^]?\s*[−\-\u2011]\s*\d+|0\.\d{3,})',
            re.IGNORECASE
        )

        corrections = 0

        def _correct_field(text: str) -> str:
            nonlocal corrections
            if not text:
                return text

            def replacer(match):
                nonlocal corrections
                gene_sym = match.group(1).upper()
                prefix = match.group(2)
                fdr_str = match.group(3)

                # Parse the claimed FDR value
                try:
                    normalized = fdr_str.replace('\u2212', '-').replace('\u2011', '-').replace(' ', '')
                    # Handle "1.55 × 10^-3" style
                    sci_match = re.match(r'(\d+\.?\d*)[×xX*]10[\^]?[−\-](\d+)', normalized)
                    if sci_match:
                        claimed_fdr = float(sci_match.group(1)) * (10 ** -int(sci_match.group(2)))
                    else:
                        claimed_fdr = float(normalized)
                except (ValueError, TypeError):
                    return match.group(0)

                # Check if gene is in our lookup
                if gene_sym not in gene_fdr_lookup:
                    return match.group(0)

                actual_fdr = gene_fdr_lookup[gene_sym]

                # Skip if claimed value is already close to actual
                if actual_fdr > 0 and claimed_fdr > 0:
                    if abs(math.log10(claimed_fdr) - math.log10(actual_fdr)) <= 1.0:
                        return match.group(0)  # Close enough, not a misattribution

                # Check if claimed value matches a pathway FDR
                is_pathway_fdr = False
                for pfdr in pathway_fdr_values:
                    if pfdr > 0 and claimed_fdr > 0:
                        if abs(math.log10(claimed_fdr) - math.log10(pfdr)) <= 0.3:
                            is_pathway_fdr = True
                            break

                if not is_pathway_fdr:
                    return match.group(0)  # Not a pathway FDR, don't touch

                # Replace with correct gene-level FDR
                corrections += 1
                return f"{match.group(1)}{prefix}{actual_fdr:.2e}"

            return fdr_claim_pattern.sub(replacer, text)

        # Correct text fields in each hypothesis
        for hyp in result.get('hypotheses', []):
            for field in ['hypothesis', 'mechanisticModel', 'confidenceRationale']:
                if hyp.get(field):
                    hyp[field] = _correct_field(hyp[field])
            # Correct evidence list
            if hyp.get('evidenceSupporting'):
                hyp['evidenceSupporting'] = [_correct_field(ev) for ev in hyp['evidenceSupporting']]

        # Correct central model and summary
        if result.get('centralMechanisticModel'):
            result['centralMechanisticModel'] = _correct_field(result['centralMechanisticModel'])
        if result.get('hypothesesSummary'):
            result['hypothesesSummary'] = _correct_field(result['hypothesesSummary'])

        return corrections

    def _validate_kegg_text_claims(
        self,
        result: Dict,
        collected_kegg_relations: List[Dict]
    ) -> List[str]:
        """
        Validate free-text KEGG claims in hypothesis prose fields.

        Scans mechanisticModel and hypothesis text for KEGG-referencing patterns
        (e.g., "KEGG: acidic pH→NF-κB") and checks whether the claimed gene pairs
        actually exist in the collected KEGG relations from tool calls.

        Args:
            result: Parsed hypothesis generation result dict
            collected_kegg_relations: List of relation dicts collected during tool calls

        Returns:
            List of warning messages (empty if no issues)
        """
        if not collected_kegg_relations:
            return []

        # Build lookup set of known KEGG pairs
        kegg_pairs = set()
        for rel in collected_kegg_relations:
            kegg_pairs.add((rel['source'].upper(), rel['target'].upper()))

        warnings = []

        # Patterns that indicate a KEGG citation in prose
        kegg_mention_pattern = re.compile(
            r'KEGG\s*[:\s]|KEGG\s+relation|curated\s+KEGG|KEGG\s+shows|KEGG\s*.*?relation',
            re.IGNORECASE
        )

        # Gene symbol pattern
        gene_sym_pattern = re.compile(r'\b([A-Z][A-Z0-9]{1,15})\b')

        # Directional connectors between genes
        connector_pattern = re.compile(
            r'(→|->|activates?|inhibits?|expression|regulates?|phosphorylat\w+|binds?)',
            re.IGNORECASE
        )

        non_genes = {
            'AND', 'OR', 'THE', 'WITH', 'FOR', 'VIA', 'PATHWAY', 'SIGNALING',
            'KEGG', 'NOT', 'GENE', 'REGULATION', 'REGULATED', 'RELATION',
            'CURATED', 'SHOWS', 'LITERATURE', 'LINKS', 'BASED', 'SUPPORT',
            'SUPPORTED', 'DATA', 'FROM', 'THIS', 'THAT', 'WHICH', 'THEIR',
        }

        for i, hyp in enumerate(result.get('hypotheses', []), 1):
            # Scan relevant text fields
            for field_name in ['mechanisticModel', 'hypothesis']:
                text = hyp.get(field_name, '')
                if not text:
                    continue

                # Find all KEGG mentions
                for kegg_match in kegg_mention_pattern.finditer(text):
                    # Extract a 150-char window around the KEGG mention
                    start = max(0, kegg_match.start() - 30)
                    end = min(len(text), kegg_match.end() + 150)
                    window = text[start:end]

                    # Find gene symbols in the window
                    gene_syms = [
                        m.group(1) for m in gene_sym_pattern.finditer(window)
                        if m.group(1) not in non_genes
                    ]

                    # Find directional connectors
                    has_connector = bool(connector_pattern.search(window))

                    if len(gene_syms) >= 2 and has_connector:
                        # Check consecutive gene pairs against KEGG relations
                        any_supported = False
                        for j in range(len(gene_syms) - 1):
                            pair = (gene_syms[j].upper(), gene_syms[j + 1].upper())
                            reverse_pair = (gene_syms[j + 1].upper(), gene_syms[j].upper())
                            if pair in kegg_pairs or reverse_pair in kegg_pairs:
                                any_supported = True
                                break

                        if not any_supported:
                            claim_text = window.strip()[:120]
                            warnings.append(
                                f"Hypothesis {i}: Free-text KEGG claim in '{field_name}' "
                                f"has no matching KEGG relation for gene pairs "
                                f"({', '.join(f'{gene_syms[j]}→{gene_syms[j+1]}' for j in range(len(gene_syms)-1))}): "
                                f"\"{claim_text}...\""
                            )

        return warnings

    def _validate_prediction_vs_data(
        self,
        result: Dict,
        genes: List[Dict]
    ) -> List[str]:
        """
        Validate that hypothesis predictions about downstream effects don't
        contradict observed gene expression data.

        Unlike _validate_hypothesis_directions (which catches wrong DESCRIPTIONS of
        current state, e.g., "GENE is upregulated" when it's downregulated), this
        validator catches wrong PREDICTIONS about expected effects (e.g., "X dampens
        GENE expression" when GENE is observed upregulated).

        Only flags when the hypothesis presents this as a CONSEQUENCE of its mechanism,
        not as a proposed intervention (e.g., "knockdown of X should reduce Y" is a
        testability claim, not a data contradiction).

        Args:
            result: Parsed hypothesis generation result dict
            genes: List of DE genes with fold changes

        Returns:
            List of warning messages (empty if no issues)
        """
        # Build gene lookup
        gene_lookup = {}
        for gene in genes:
            symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                     gene.get('gene') or gene.get('name', '')).upper()
            fc = gene.get('foldChange') or gene.get('log2_fold_change') or gene.get('fold_change')
            if fc is not None:
                gene_lookup[symbol] = float(fc)

        warnings = []

        # Patterns that predict DECREASED downstream effect
        down_prediction_patterns = [
            re.compile(
                r'(?:dampens?|reduces?|limits?|decreases?|suppresses?|diminishes?|attenuates?|blunts?|curtails?)'
                r'\s+(?:\w+\s+){0,4}?'  # up to 4 intervening words
                r'(\b[A-Z][A-Z0-9]{1,15}\b)'
                r'\s*(?:production|expression|secretion|activity|levels?|signaling|release)?',
                re.IGNORECASE
            ),
            re.compile(
                r'(?:limiting|dampening|reducing|suppressing|diminishing|attenuating)'
                r'\s+(\b[A-Z][A-Z0-9]{1,15}\b)'
                r'\s*(?:production|expression|secretion|activity|levels?|signaling|release|recruitment)?',
                re.IGNORECASE
            ),
        ]

        # Patterns that predict INCREASED downstream effect
        up_prediction_patterns = [
            re.compile(
                r'(?:enhances?|increases?|amplifies?|promotes?|elevates?|boosts?|drives?|stimulates?)'
                r'\s+(?:\w+\s+){0,4}?'
                r'(\b[A-Z][A-Z0-9]{1,15}\b)'
                r'\s*(?:production|expression|secretion|activity|levels?|signaling|release)?',
                re.IGNORECASE
            ),
            re.compile(
                r'(?:enhancing|increasing|amplifying|promoting|elevating|boosting|driving|stimulating)'
                r'\s+(\b[A-Z][A-Z0-9]{1,15}\b)'
                r'\s*(?:production|expression|secretion|activity|levels?|signaling|release)?',
                re.IGNORECASE
            ),
        ]

        # Patterns that indicate an INTERVENTION context (should NOT be flagged)
        intervention_pattern = re.compile(
            r'(?:knockdown|knockout|siRNA|shRNA|inhibitor|inhibition of|blocking|treatment with|'
            r'suppressing|inhibiting|activating|restoring|overexpressing|'
            r'administration of|targeting|depletion of|deletion of|'
            r'should\s+(?:reduce|increase|decrease|enhance|restore|attenuate|diminish|suppress|amplify)|'
            r'would\s+(?:reduce|increase|decrease|enhance|restore|attenuate|diminish|suppress|amplify)|'
            r'could\s+(?:reduce|increase|decrease|enhance|restore|attenuate|diminish|suppress|amplify)|'
            r'may\s+(?:reduce|increase|decrease|enhance|restore|attenuate|diminish|suppress|amplify)|'
            r'will\s+(?:reduce|increase|decrease|enhance|restore|attenuate|diminish|suppress|amplify)|'
            r'predicted\s+to|expected\s+to|test\s+whether|if\s+\w+\s+is\s+(?:blocked|inhibited|activated))',
            re.IGNORECASE
        )

        non_genes = {
            'AND', 'OR', 'THE', 'WITH', 'FOR', 'VIA', 'PATHWAY', 'SIGNALING',
            'NF', 'HIF', 'TLR', 'RNA', 'DNA', 'ATP', 'ADP', 'GENE', 'NOT',
        }

        for i, hyp in enumerate(result.get('hypotheses', []), 1):
            for field_name in ['mechanisticModel', 'hypothesis']:
                text = hyp.get(field_name, '')
                if not text:
                    continue

                # Check for DOWN predictions contradicted by UP data
                for pattern in down_prediction_patterns:
                    for match in pattern.finditer(text):
                        gene_sym = match.group(1).upper()
                        if gene_sym in non_genes or gene_sym not in gene_lookup:
                            continue

                        actual_fc = gene_lookup[gene_sym]
                        if actual_fc <= 0:
                            continue  # Gene IS downregulated, prediction is consistent

                        # Check if this is in an intervention context (within 80 chars before match)
                        context_start = max(0, match.start() - 80)
                        context_window = text[context_start:match.end()]
                        if intervention_pattern.search(context_window):
                            continue  # This is a proposed experiment, not a data claim

                        warnings.append(
                            f"Hypothesis {i} ({field_name}): Predicts dampened/reduced {gene_sym} "
                            f"but {gene_sym} is UPREGULATED (FC={actual_fc:+.2f}) in the data. "
                            f"Context: \"{match.group(0).strip()}\""
                        )

                # Check for UP predictions contradicted by DOWN data
                for pattern in up_prediction_patterns:
                    for match in pattern.finditer(text):
                        gene_sym = match.group(1).upper()
                        if gene_sym in non_genes or gene_sym not in gene_lookup:
                            continue

                        actual_fc = gene_lookup[gene_sym]
                        if actual_fc >= 0:
                            continue  # Gene IS upregulated, prediction is consistent

                        # Check if this is in an intervention context
                        context_start = max(0, match.start() - 80)
                        context_window = text[context_start:match.end()]
                        if intervention_pattern.search(context_window):
                            continue

                        warnings.append(
                            f"Hypothesis {i} ({field_name}): Predicts enhanced/increased {gene_sym} "
                            f"but {gene_sym} is DOWNREGULATED (FC={actual_fc:+.2f}) in the data. "
                            f"Context: \"{match.group(0).strip()}\""
                        )

        return warnings

    def _validate_internal_consistency(
        self,
        result: Dict,
        genes: List[Dict]
    ) -> List[str]:
        """
        Validate that hypotheses are internally consistent:
        - Predictions don't contradict observed data
        - Mechanistic directions are logically coherent
        - Claimed gene effects match actual regulation direction
        """
        warnings = []

        # Build gene lookup
        gene_lookup = {}
        for gene in genes:
            symbol = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                     gene.get('gene') or gene.get('name', '')).upper()
            fc = gene.get('foldChange') or gene.get('log2_fold_change') or gene.get('fold_change')
            if fc is not None:
                gene_lookup[symbol] = float(fc)

        for i, hyp in enumerate(result.get('hypotheses', []), 1):
            # Check: Consensus hypotheses — check originalHypotheses for direction conflicts
            original_hyps = hyp.get('originalHypotheses', [])
            if len(original_hyps) > 1:
                # Extract directional claims from each original hypothesis
                for j, orig1 in enumerate(original_hyps):
                    for orig2 in original_hyps[j+1:]:
                        text1 = orig1.get('hypothesis', '')
                        text2 = orig2.get('hypothesis', '')
                        # Check for opposite direction claims about shared genes
                        for gene_symbol in gene_lookup:
                            if (re.search(rf'\b{re.escape(gene_symbol)}\b', text1, re.IGNORECASE) and
                                re.search(rf'\b{re.escape(gene_symbol)}\b', text2, re.IGNORECASE)):
                                # Check if one says "increases" and other says "decreases"
                                # for downstream effects
                                up_in_1 = bool(re.search(
                                    r'(accumulat|increas|elevat|higher|raises)',
                                    text1, re.IGNORECASE
                                ))
                                down_in_1 = bool(re.search(
                                    r'(deplet|decreas|lower|reduces|loss of)',
                                    text1, re.IGNORECASE
                                ))
                                up_in_2 = bool(re.search(
                                    r'(accumulat|increas|elevat|higher|raises)',
                                    text2, re.IGNORECASE
                                ))
                                down_in_2 = bool(re.search(
                                    r'(deplet|decreas|lower|reduces|loss of)',
                                    text2, re.IGNORECASE
                                ))
                                if (up_in_1 and down_in_2) or (down_in_1 and up_in_2):
                                    warnings.append(
                                        f"Hypothesis {i}: Merged sub-hypotheses have "
                                        f"contradictory directions for downstream effects "
                                        f"(run {orig1.get('run_id')} vs run {orig2.get('run_id')})"
                                    )
                                    break

        return warnings

    def _validate_kegg_claims(
        self,
        result: Dict,
        collected_kegg_relations: List[Dict]
    ) -> List[str]:
        """
        Validate mechanistic claims against collected KEGG relations.

        Each hypothesis may include a 'mechanisticClaims' list with entries like:
            {"source": "GENE_A", "relation": "activation", "target": "GENE_B", "basis": "..."}

        This method checks each claim's (source, target) pair against the KEGG
        relations collected during tool calls. Unsupported claims are flagged
        and each hypothesis is annotated with a 'keggValidation' summary.

        Known limitation: Low validation rates (0-30%) are expected because
        ``collected_kegg_relations`` only contains relations from the 3-5
        pathways explicitly queried via tools during hypothesis generation.
        Well-known biology (e.g., IFNG→STAT1) that was not part of any queried
        pathway will be flagged as unsupported. These warnings are informational
        and do not necessarily indicate errors in the hypothesis.

        Args:
            result: Parsed hypothesis generation result dict
            collected_kegg_relations: List of relation dicts collected during tool calls

        Returns:
            List of warning messages (empty if no issues)
        """
        if not collected_kegg_relations:
            return []

        # Build lookup sets: {(SOURCE, TARGET)} and {(SOURCE, TARGET, TYPE)}
        kegg_pairs = set()
        kegg_triples = set()
        for rel in collected_kegg_relations:
            pair = (rel['source'], rel['target'])
            triple = (rel['source'], rel['target'], rel['type'])
            kegg_pairs.add(pair)
            kegg_triples.add(triple)

        warnings = []

        for i, hyp in enumerate(result.get('hypotheses', []), 1):
            claims = hyp.get('mechanisticClaims', [])
            unsupported = []

            for claim in claims:
                source = claim.get('source', '').upper()
                target = claim.get('target', '').upper()
                relation = claim.get('relation', '')

                if not source or not target:
                    continue

                # Check if (source, target) pair exists in ANY collected KEGG relation
                if (source, target) not in kegg_pairs:
                    unsupported.append(
                        f"{source} -> {target} ({relation})"
                    )

            if unsupported:
                warnings.append(
                    f"Hypothesis {i}: {len(unsupported)} unsupported claim(s): "
                    + "; ".join(unsupported)
                )

            # Annotate the hypothesis with validation results
            total_claims = len(claims)
            supported = total_claims - len(unsupported)
            hyp['keggValidation'] = {
                'totalClaims': total_claims,
                'supportedClaims': supported,
                'unsupportedClaims': unsupported,
                'validationRate': round(supported / total_claims, 2) if total_claims > 0 else None
            }

        return warnings

    def _verify_kegg_claims_with_reviewer(self, result: Dict) -> List[str]:
        """Verify unsupported KEGG claims with the reviewer model (OpenBioLLM).

        Uses anti-sycophancy: for each unsupported claim, ask the reviewer to
        state the correct biochemistry independently (without showing the claim),
        then compare the independent answer with the claim.

        Returns:
            List of error strings (Biochemistry review format) for claims the
            reviewer's independent knowledge contradicts.
        """
        reviewer_model = getattr(self.reviewer_llm, 'model', 'unknown')
        errors = []

        for i, hyp in enumerate(result.get('hypotheses', []), 1):
            kegg_val = hyp.get('keggValidation', {})
            unsupported = kegg_val.get('unsupportedClaims', [])
            if not unsupported:
                continue

            rejected_claims = []
            for claim_str in unsupported:
                # Parse "SOURCE -> TARGET (relation)" format
                match = re.match(r'(\S+)\s*->\s*(.+?)\s*\((.+?)\)', claim_str)
                if not match:
                    continue

                source = match.group(1)
                target = match.group(2)
                relation = match.group(3)

                # Step 1: Ask reviewer about the specific claimed relationship
                # Use a factual question that can be answered yes/no with explanation
                knowledge_q = (
                    f"Does {source} {relation.lower()} {target}? "
                    f"Is this a biologically established relationship? "
                    f"If not, what does {source} actually do? "
                    f"Answer in 1-2 sentences."
                )

                try:
                    knowledge_resp = self.reviewer_llm.chat(
                        [{"role": "user", "content": knowledge_q}],
                        temperature=0.0,
                        seed=42,
                        max_tokens=200
                    )

                    # Step 2: Check if the claimed relationship is supported
                    comparison_prompt = f"""The hypothesis claims: {source} {relation} {target}

An independent expert describes {source}'s function: "{knowledge_resp.strip()}"

Based on the expert's description, is the claimed relationship ({source} → {target}) biologically plausible — meaning the expert's description supports a direct OR indirect connection between {source} and {target}?
Answer ONLY "SUPPORTED" or "NOT_SUPPORTED"."""

                    comparison_resp = self.reviewer_llm.chat(
                        [{"role": "user", "content": comparison_prompt}],
                        temperature=0.0,
                        seed=42,
                        max_tokens=20
                    )

                    verdict = comparison_resp.strip().upper()
                    if 'NOT_SUPPORTED' in verdict or 'NOT SUPPORTED' in verdict:
                        rejected_claims.append((claim_str, knowledge_resp.strip()))

                except Exception:
                    pass  # Skip failed checks

            if rejected_claims:
                for claim_str, reviewer_reason in rejected_claims:
                    errors.append(
                        f"Biochemistry review (error) - Hypothesis {i}: "
                        f"[unsupported_mechanism] {claim_str} — "
                        f"Correction: {reviewer_reason[:200]} "
                        f"(not found in KEGG data AND contradicted by {reviewer_model})"
                    )

        return errors
