"""
Baseline Comparison Service

Sends the same input data (DE genes, enriched pathways, experimental context) to
external LLMs (OpenAI GPT, Anthropic Claude) and asks them to generate mechanistic
hypotheses, providing baselines for comparing against the pipeline's outputs.
"""

import json
import re
import time
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.config import settings
from src.pipeline.fc_utils import fc_arrow, to_float


class BaselineComparisonService:
    """Runs external LLMs as baseline hypothesis generators for comparison."""

    PROVIDERS = {
        'openai': {
            'api_key_attr': 'OPENAI_API_KEY',
            'model_attr': 'OPENAI_BASELINE_MODEL',
            'url_attr': 'OPENAI_API_URL',
        },
        'anthropic': {
            'api_key_attr': 'ANTHROPIC_API_KEY',
            'model_attr': 'ANTHROPIC_BASELINE_MODEL',
            'url_attr': 'ANTHROPIC_API_URL',
        },
    }

    def __init__(self, provider: str = 'openai'):
        """
        Initialize baseline comparison service.

        Args:
            provider: 'openai' or 'anthropic'
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'")

        self.provider = provider
        cfg = self.PROVIDERS[provider]
        self.api_key = getattr(settings, cfg['api_key_attr'], '')
        self.model = getattr(settings, cfg['model_attr'], '')
        self.api_url = getattr(settings, cfg['url_attr'], '')

    def run_baseline(
        self,
        genes: List[Dict],
        pathways: List[Dict],
        context: Optional[Dict],
        experiment_context: str,
        output_dir: Path
    ) -> Optional[Dict]:
        """
        Generate baseline hypotheses using the configured LLM.

        Args:
            genes: List of DE genes with fold changes and p-values
            pathways: List of enriched pathways
            context: Experimental context dict
            experiment_context: Human-readable context string
            output_dir: Directory to save baseline output

        Returns:
            Dict with baseline hypotheses, or None on failure
        """
        if not self.api_key:
            print(f'  Baseline comparison ({self.provider}): No API key configured, skipping')
            return None

        print(f'\n{"="*80}')
        print(f'BASELINE COMPARISON: {self.model} ({self.provider})')
        print(f'{"="*80}')
        print(f'  Model: {self.model}')
        print(f'  Endpoint: {self.api_url}')

        prompt = self._build_prompt(genes, pathways, experiment_context)
        prompt_tokens_est = len(prompt) // 4
        print(f'  Prompt size: ~{prompt_tokens_est} tokens (est)')

        # Run 3 times for reproducibility comparison
        all_runs = []
        for run_idx in range(1, 4):
            print(f'\n  --- Baseline Run {run_idx}/3 ---')
            start_time = time.time()

            try:
                result = self._call_llm(prompt)
                elapsed = time.time() - start_time
                print(f'  Response received: {len(result)} chars ({elapsed:.1f}s)')

                parsed = self._parse_response(result)
                if parsed and parsed.get('hypotheses'):
                    num_hyps = len(parsed['hypotheses'])
                    print(f'  Hypotheses generated: {num_hyps}')
                    for i, h in enumerate(parsed['hypotheses'], 1):
                        conf = h.get('confidence', '?')
                        stmt = h.get('hypothesis', '')[:100]
                        print(f'    {run_idx}.{i} [{conf}]: {stmt}...')
                    all_runs.append({
                        'run': run_idx,
                        'hypotheses': parsed['hypotheses'],
                        'central_model': parsed.get('centralMechanisticModel', ''),
                        'elapsed_seconds': round(elapsed, 1)
                    })
                else:
                    print(f'  Failed to parse hypotheses from response')
                    all_runs.append({
                        'run': run_idx,
                        'hypotheses': [],
                        'parse_error': True,
                        'raw_response': result[:2000],
                        'elapsed_seconds': round(elapsed, 1)
                    })

            except Exception as e:
                elapsed = time.time() - start_time
                print(f'  Error: {e} ({elapsed:.1f}s)')
                all_runs.append({
                    'run': run_idx,
                    'hypotheses': [],
                    'error': str(e),
                    'elapsed_seconds': round(elapsed, 1)
                })

        # Compile results
        baseline_result = {
            'provider': self.provider,
            'model': self.model,
            'num_runs': 3,
            'runs': all_runs,
            'experiment_context': experiment_context,
            'num_genes_provided': len(genes),
            'num_pathways_provided': len(pathways),
            'prompt_tokens_est': prompt_tokens_est,
        }

        # Save to output directory
        output_file = output_dir / f'baseline_{self.provider}.json'
        with open(output_file, 'w') as f:
            json.dump(baseline_result, f, indent=2)
        print(f'\n  Baseline results saved to: {output_file}')

        self._print_summary(all_runs)
        return baseline_result

    def _build_prompt(
        self,
        genes: List[Dict],
        pathways: List[Dict],
        experiment_context: str
    ) -> str:
        """Build the prompt with the same input data the pipeline receives."""

        sorted_genes = sorted(
            genes,
            key=lambda g: abs(g.get('foldChange', g.get('log2_fold_change', 0))),
            reverse=True
        )

        gene_lines = []
        for g in sorted_genes[:50]:
            symbol = g.get('geneSymbol', g.get('gene_symbol', ''))
            fc = to_float(g.get('foldChange', g.get('log2_fold_change', 0))) or 0.0
            pval = g.get('pValue', g.get('p_value', 0))
            fdr = g.get('pValueFDR', g.get('p_value_fdr', 0))
            arrow = fc_arrow(fc)
            gene_lines.append(
                f"  {symbol}: FC={fc:+.2f}{' ' + arrow if arrow else ''}, p={pval:.2e}, FDR={fdr:.2e}"
            )

        pathway_lines = []
        for p in pathways:
            name = p.get('name', '')
            es = p.get('ES', 0)
            fdr = p.get('pValueFDR', p.get('pValue', 0))
            p_genes = p.get('genes', [])
            direction = 'UP' if es > 0 else 'DOWN'
            gene_str = ', '.join(p_genes[:10])
            if len(p_genes) > 10:
                gene_str += f'... (+{len(p_genes)-10} more)'
            pathway_lines.append(
                f"  {name}: ES={es:+.2f} ({direction}), FDR={fdr:.2e}, genes=[{gene_str}]"
            )

        prompt = f"""You are a molecular biology expert. Based on the following differential expression and pathway enrichment data from a {experiment_context} study, generate 5-7 testable mechanistic hypotheses.

**Experimental Context:** {experiment_context}
**Total DE genes:** {len(genes)} (showing top 50 by |fold change|)
**Enriched KEGG pathways:** {len(pathways)}

**Top Differentially Expressed Genes (log2 fold change):**
{chr(10).join(gene_lines)}

**Enriched KEGG Pathways:**
{chr(10).join(pathway_lines)}

**Instructions:**
1. Generate 5-7 testable mechanistic hypotheses that explain the observed molecular changes
2. Each hypothesis should connect multiple genes and pathways mechanistically
3. Use SPECIFIC gene names and fold changes from the data above
4. Make DIRECTIONAL predictions grounded in observed data
5. Propose CONCRETE experimental tests for each hypothesis
6. Assign confidence: high, medium, or low

**CRITICAL RULES:**
- Fold change > 0 = UP-regulated, Fold change < 0 = DOWN-regulated
- Do NOT assume pathway enrichment means gene upregulation
- Only cite genes and statistics from the data provided above
- If an enzyme is downregulated, its product DECREASES and substrate ACCUMULATES

Return ONLY valid JSON in this format:
{{
  "hypotheses": [
    {{
      "hypothesis": "Clear, testable hypothesis statement",
      "mechanisticModel": "Detailed molecular mechanism (3-4 sentences)",
      "keyPlayers": ["GENE1 - role", "GENE2 - role"],
      "evidenceSupporting": ["Evidence 1", "Evidence 2"],
      "testability": {{
        "approach1": "Experimental test 1",
        "approach2": "Experimental test 2",
        "expectedOutcome": "Directional prediction"
      }},
      "confidence": "high|medium|low"
    }}
  ],
  "centralMechanisticModel": "Overall unifying model (4-5 sentences)"
}}"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM API."""
        if self.provider == 'openai':
            return self._call_openai(prompt)
        elif self.provider == 'anthropic':
            return self._call_anthropic(prompt)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            # "temperature": 0,
            "max_completion_tokens": 8000,
            # "seed": 42
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=180
        )
        response.raise_for_status()

        data = response.json()
        return data['choices'][0]['message']['content']

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": 16000,
            # "temperature": 0,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        data = response.json()
        stop_reason = data.get('stop_reason', 'unknown')
        if stop_reason == 'max_tokens':
            print(f'  ⚠️  Response truncated (hit max_tokens limit)')

        content_blocks = data.get('content', [])
        text_parts = [block['text'] for block in content_blocks if block.get('type') == 'text']
        return ''.join(text_parts)

    def _parse_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        text = response.strip()

        # Handle markdown code blocks — extract content between ``` markers
        if '```' in text:
            parts = text.split('```')
            for part in parts[1:]:  # Skip text before first ```
                cleaned = part.strip()
                if cleaned.startswith('json'):
                    cleaned = cleaned[4:].strip()
                # Try to parse this block
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Regex fallback — find outermost { ... }
        # Use a balanced brace approach for large JSON
        start = text.find('{')
        if start >= 0:
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

        return None

    def _print_summary(self, all_runs: List[Dict]):
        """Print comparison summary across runs."""
        print(f'\n  {"="*60}')
        print(f'  BASELINE SUMMARY ({self.model})')
        print(f'  {"="*60}')

        for run in all_runs:
            run_idx = run['run']
            num = len(run.get('hypotheses', []))
            elapsed = run.get('elapsed_seconds', 0)
            status = f'{num} hypotheses' if num > 0 else 'FAILED'
            print(f'    Run {run_idx}: {status} ({elapsed}s)')

        if all(len(r.get('hypotheses', [])) > 0 for r in all_runs):
            all_themes = []
            for run in all_runs:
                themes = set()
                for h in run['hypotheses']:
                    players = h.get('keyPlayers', [])
                    genes = set()
                    for p in players:
                        found = re.findall(r'\b([A-Z][A-Z0-9]{1,10})\b', p.split('–')[0].split('-')[0].split('(')[0])
                        genes.update(found)
                    themes.add(frozenset(genes))
                all_themes.append(themes)

            if len(all_themes) >= 2:
                run1_genes = set().union(*[t for t in all_themes[0]])
                run2_genes = set().union(*[t for t in all_themes[1]])
                overlap = run1_genes & run2_genes
                union = run1_genes | run2_genes
                jaccard = len(overlap) / len(union) if union else 0
                print(f'\n    Gene overlap between Run 1 & 2: {jaccard:.2f} Jaccard')
                print(f'    Shared genes: {sorted(overlap)[:10]}')
