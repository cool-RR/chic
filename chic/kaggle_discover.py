#!/usr/bin/env python
"""
Kaggle Model Discovery Tool
Helps find and list Flax model versions from Kaggle
"""

from __future__ import annotations

import sys
from typing import Optional

import click
import kagglehub
from kagglehub.clients import KaggleApiV1Client


class Color:
    """Color codes - automatically disabled when stdout is not a TTY."""

    @staticmethod
    def _should_use_color() -> bool:
        """Check if we should use colors based on TTY status."""
        return sys.stdout.isatty()

    @classmethod
    def _get(cls, code: str) -> str:
        """Get color code if TTY, otherwise empty string."""
        return code if cls._should_use_color() else ''

    @property
    def BOLD(self) -> str:
        return self._get('\033[1m')

    @property
    def GREEN(self) -> str:
        return self._get('\033[92m')

    @property
    def RED(self) -> str:
        return self._get('\033[91m')

    @property
    def YELLOW(self) -> str:
        return self._get('\033[93m')

    @property
    def BLUE(self) -> str:
        return self._get('\033[94m')

    @property
    def CYAN(self) -> str:
        return self._get('\033[96m')

    @property
    def MAGENTA(self) -> str:
        return self._get('\033[95m')

    @property
    def END(self) -> str:
        return self._get('\033[0m')


# Create a singleton instance to use throughout the module
color = Color()


def search_models_by_framework(query: str, framework_filters: tuple[str, ...] = ("flax", "jax"), debug: bool = False) -> list[dict]:
    """
    Search for models on Kaggle using the KaggleApiV1Client.

    Args:
        query: Search term (e.g., 'gemma', 'llama', 'qwen')
        framework_filters: Tuple of framework names to filter by
        debug: Print debug information

    Returns:
        List of model dictionaries with metadata including handle, framework, owner, model, variation
    """
    api = KaggleApiV1Client()

    # Try multiple possible API endpoint paths
    possible_paths = [
        f"models/search?query={query}&page_size=200",
        f"models/list?search={query}&page_size=200",
        f"models/list?query={query}&page_size=200",
    ]

    json_resp = None
    last_error = None
    for path in possible_paths:
        try:
            if debug:
                print(f"Trying API path: {path}")
            json_resp = api.get(path)
            if debug:
                print(f"Response type: {type(json_resp)}")
                if isinstance(json_resp, dict):
                    print(f"Response keys: {list(json_resp.keys())}")
            if json_resp:
                break
        except Exception as e:
            last_error = e
            if debug:
                print(f"Error on path {path}: {e}")
            continue

    if json_resp is None:
        error_msg = "Could not find a suitable models list/search endpoint on the Kaggle server."
        if last_error and debug:
            error_msg += f" Last error: {last_error}"
        raise RuntimeError(error_msg)

    # Parse the response - try different common structures
    candidates = []
    if isinstance(json_resp, dict):
        # Try common keys for model lists
        for key in ("models", "items", "results"):
            if key in json_resp and isinstance(json_resp[key], list):
                candidates = json_resp[key]
                if debug:
                    print(f"Found {len(candidates)} candidates in '{key}' field")
                    if candidates and len(candidates) > 0:
                        print(f"First candidate keys: {list(candidates[0].keys())}")
                break
        else:
            # Maybe it's a single model dict
            if isinstance(json_resp.get("owner"), str) and json_resp.get("slug"):
                candidates = [json_resp]
            else:
                # Fallback: find any list value
                for v in json_resp.values():
                    if isinstance(v, list):
                        candidates = v
                        break
    elif isinstance(json_resp, list):
        candidates = json_resp

    results = []
    seen_handles = set()

    for item in candidates:
        # Get owner and model name from the top level
        # The API returns 'author' and 'slug' at the top level
        owner = item.get("author") or item.get("owner") or item.get("Owner")
        model_slug = item.get("slug") or item.get("model")

        if not owner or not model_slug:
            if debug:
                print(f"Skipping item - missing owner or slug: {item.get('title', 'unknown')}")
            continue

        # Process instances (framework/variation combinations)
        instances = item.get("instances", [])
        if not instances:
            if debug:
                print(f"No instances found for {owner}/{model_slug}")
            continue

        if debug:
            print(f"Processing {owner}/{model_slug} with {len(instances)} instances")

        for inst in instances:
            # Get framework and variation from instance
            framework = inst.get("framework") or inst.get("Framework")
            variation = inst.get("slug") or inst.get("instanceSlug") or inst.get("variation")

            if not framework or not variation:
                continue

            # Filter by framework
            fw_lower = framework.lower()
            if not any(f in fw_lower for f in framework_filters):
                continue

            # Build the full Kaggle handle
            handle = f"{owner}/{model_slug}/{framework}/{variation}"

            if handle not in seen_handles:
                seen_handles.add(handle)
                results.append({
                    'handle': handle,
                    'owner': owner,
                    'model': model_slug,
                    'framework': framework,
                    'variation': variation,
                    'url': f'https://www.kaggle.com/models/{owner}/{model_slug}',
                    'title': item.get('title', model_slug)
                })

                if debug:
                    print(f"  Added: {handle}")

    return sorted(results, key=lambda x: x['handle'])


@click.command()
@click.argument('search_term', required=False, default=None)
@click.option('--framework', '-f', type=click.Choice(['flax', 'jax', 'pytorch', 'tensorflow'],
              case_sensitive=False), default=None,
              help='Filter by framework (flax, jax, pytorch, tensorflow)')
@click.option('--format', '-o', 'output_format',
              type=click.Choice(['simple', 'detailed', 'python'], case_sensitive=False),
              default='detailed',
              help='Output format: simple (handles only), detailed (with info), python (Python code)')
@click.option('--debug', is_flag=True, help='Show debug information')
def main(search_term: Optional[str], framework: Optional[str], output_format: str, debug: bool) -> None:
    """
    Discover and list Kaggle models, especially Flax versions.

    Examples:

        # Search for Gemma models (defaults to flax/jax)
        chic kaggle-discover gemma

        # Search for Gemma Flax models only
        chic kaggle-discover gemma --framework flax

        # Search for all PyTorch models
        chic kaggle-discover llama --framework pytorch

        # Get Python code for model definitions
        chic kaggle-discover qwen --framework flax --format python
    """

    # Default searches if no term provided
    if search_term is None:
        print(f"{color.CYAN}Common model families to search:{color.END}")
        print("  - gemma")
        print("  - llama")
        print("  - qwen")
        print("  - mistral")
        print("  - phi")
        print()
        print("Run with a search term, e.g.: chic kaggle-discover gemma")
        return

    print(f"{color.BOLD}Searching Kaggle for: {search_term}{color.END}")

    # Determine framework filters
    if framework:
        framework_filters = (framework.lower(),)
        print(f"Framework filter: {framework}")
    else:
        # Default to flax/jax
        framework_filters = ("flax", "jax")
        print(f"Framework filter: flax, jax (default)")
    print()

    try:
        models = search_models_by_framework(search_term, framework_filters, debug=debug)
    except Exception as e:
        print(f"{color.RED}Error searching models: {e}{color.END}")
        print(f"{color.YELLOW}Tip: Make sure you have Kaggle credentials configured.{color.END}")
        print(f"{color.YELLOW}Run: kagglehub.login() or set KAGGLE_USERNAME and KAGGLE_KEY{color.END}")
        return

    if not models:
        print(f"{color.YELLOW}No models found matching '{search_term}'")
        if framework:
            print(f"with framework '{framework}'")
        print(f"{color.END}")
        return

    print(f"{color.GREEN}Found {len(models)} model(s):{color.END}\n")

    if output_format == 'simple':
        # Simple format: just handles
        for model in models:
            print(model['handle'])

    elif output_format == 'detailed':
        # Detailed format with colors and information
        for i, model in enumerate(models, 1):
            print(f"{color.BOLD}{i}. {model['handle']}{color.END}")
            print(f"   Owner: {color.CYAN}{model['owner']}{color.END}")
            print(f"   Model: {color.CYAN}{model['model']}{color.END}")
            print(f"   Framework: {color.CYAN}{model['framework']}{color.END}")
            print(f"   Variation: {color.CYAN}{model['variation']}{color.END}")
            print(f"   URL: {color.BLUE}{model['url']}{color.END}")
            print(f"   Download: {color.MAGENTA}kagglehub.model_download('{model['handle']}'){color.END}")
            print()

    elif output_format == 'python':
        # Python code format for easy copy-paste into model_brand.py
        print(f"{color.YELLOW}# Python code for model_brand.py:{color.END}\n")

        for model in models:
            owner = model['owner']
            model_name = model['model']
            framework = model['framework']
            variation = model['variation']

            # Generate Python code
            print(f"# {model['handle']}")
            print(f"ModelBrand(")
            print(f"    name='{variation}',  # Update this name")
            print(f"    model_family=ModelFamily.{model_name.upper().replace('-', '_')},  # Add to enum if needed")
            print(f"    kaggle_path='{owner}/{model_name}/{framework}/',")
            print(f"),")
            print()

    # Additional tips
    if not framework:
        print(f"{color.YELLOW}Tip: By default, only flax/jax models are shown.{color.END}")
        print(f"{color.YELLOW}     Use --framework to filter by other frameworks.{color.END}")


if __name__ == "__main__":
    main()
