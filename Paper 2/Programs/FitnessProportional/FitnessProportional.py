"""Grammatical evolution with sampled fitness-proportional selection.

This experiment isolates the Paper 2 selection operator while retaining the
original core fitness function.  It evolves grammars
over a configurable fixed set of nonterminals and the terminals a, b, c,
using N1 as the start symbol.

Example:
    python "[path to file]" "[path to data]" [optionals]
        --generations num                       : Num of generations. Default: 1000
        --seed num                              : Seed for randomization
        --nonterminals num                      : Fixed symbols N1 through N. Default: 4
        --language-sample-limit num             : Sets the number of generated strings used for
                                                  Levenshtein smoothing. Default: 256
        --target-grammar "[path\\target.json]"   : Stop on bounded target-language equivalence
        --target-equivalence-interval num        : Check every N generations. Default: 25
        --graph                                 : Generates a single graph with average and best
                                                  fitness over generations
        --graphs                                : Generates --graph and a second graph comparing 5
                                                  trials' bests
        --graph-file "[path\\graph.png]"        : Updates where --graph puts the graph
        --graph-directory "[path\\Graphs]"      : Updates where --graphs puts the graphs
        --quiet                                 : Hides generation updates
        --target-grammar "path\to\target.json"  : Target grammar for checking equivelance
        --target-equivalence-interval num       : Interval of generations in which it checks 
                                                  the semantic equivelance
MAX_LANGUAGE_LENGTH must match the longest string length supported by the data.

The workbook must contain the columns ``Generated String`` and
``Production Rule Applications``.  Python 3.10+, NumPy, and Matplotlib are required;
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import numpy as np


TERMINALS = ("a", "b", "c")
DEFAULT_NONTERMINAL_COUNT = 4
NONTERMINALS = tuple(f"N{index}" for index in range(1, DEFAULT_NONTERMINAL_COUNT + 1))
START_SYMBOL = NONTERMINALS[0]

POPULATION_SIZE = 100
SELECTION_SAMPLE_SIZE = 10
ELITE_COUNT = 4
DEFAULT_GENERATIONS = 1000
TRIAL_COUNT = 5

C1 = 100.0
C2 = 80.0
C3 = 2.0
C4 = 0.1
MAX_LANGUAGE_LENGTH = 30  # Must be at least the longest string in the selected data.
LANGUAGE_CELL_LIMIT = 1024
TARGET_LANGUAGE_STORAGE_LIMIT = 10_000
DEFAULT_TARGET_EQUIVALENCE_INTERVAL = 25

# Fitness still performs exact membership tests for every training example.
# This bound applies only to the representative language used for the smooth
# Levenshtein component and prevents highly permissive grammars from exhausting
# memory by materializing all 3^14 possible strings.
DEFAULT_LANGUAGE_SAMPLE_LIMIT = 256


@dataclass(frozen=True)
class GrammarConfig:
    """The fixed symbol vocabulary used throughout one evolutionary run."""

    nonterminals: tuple[str, ...]
    terminals: tuple[str, ...] = TERMINALS
    start_symbol: str = START_SYMBOL

    def __post_init__(self) -> None:
        if len(self.nonterminals) < 2:
            raise ValueError("At least two nonterminals are required.")
        if len(set(self.nonterminals)) != len(self.nonterminals):
            raise ValueError("Nonterminal names must be unique.")
        if not self.terminals or len(set(self.terminals)) != len(self.terminals):
            raise ValueError("At least one unique terminal is required.")
        if self.start_symbol not in self.nonterminals:
            raise ValueError("The start symbol must be one of the nonterminals.")
        if set(self.nonterminals) & set(self.terminals):
            raise ValueError("Terminal and nonterminal symbols must be disjoint.")

    @classmethod
    def from_count(cls, count: int) -> "GrammarConfig":
        if count < 2:
            raise ValueError("nonterminal count must be at least 2.")
        nonterminals = tuple(f"N{index}" for index in range(1, count + 1))
        return cls(nonterminals, TERMINALS, nonterminals[0])


DEFAULT_GRAMMAR_CONFIG = GrammarConfig(NONTERMINALS, TERMINALS, START_SYMBOL)


@dataclass(frozen=True, order=True)
class Rule:
    """One CNF production: A -> BC or A -> a."""

    lhs: str
    first: str
    second: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.second is None

    def __str__(self) -> str:
        rhs = self.first if self.is_terminal else f"{self.first} {self.second}"
        return f"{self.lhs} -> {rhs}"


@dataclass(frozen=True)
class TrainingExample:
    text: str
    applications: int


@dataclass
class Individual:
    rules: frozenset[Rule]
    fitness: float = float("-inf")

    def copy(self) -> "Individual":
        return Individual(self.rules, self.fitness)


@dataclass(frozen=True)
class LanguageProfile:
    """A representative bounded language and a safe distinct-size bound."""

    strings: tuple[str, ...]
    size_lower_bound: int
    exact_size: bool


@dataclass(frozen=True)
class FitnessBreakdown:
    fitness: float
    recall: float
    precision: float
    similarity: float
    matched_examples: int
    language_size_lower_bound: int
    language_size_exact: bool
    rule_count: int


@dataclass
class RunResult:
    best: Individual
    completed_generations: int
    average_fitness: list[float]
    best_fitness: list[float]
    best_breakdown: FitnessBreakdown | None = None
    termination_reason: str = "generation_limit"
    target_equivalent: Individual | None = None


@dataclass(frozen=True)
class TargetGrammar:
    """A validated target grammar and its exact language through one bound."""

    rules: frozenset[Rule]
    max_length: int
    language: tuple[str, ...]


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"Invalid Excel cell reference: {reference!r}")
    result = 0
    for character in letters.group(0):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _xlsx_rows(filename: Path, sheet_name: str = "Training Data") -> list[list[object | None]]:
    """Read values from one worksheet using only the Python standard library."""
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relationship_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_ns = "http://schemas.openxmlformats.org/package/2006/relationships"

    with zipfile.ZipFile(filename) as archive:
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook_root.find(f"{{{main_ns}}}sheets")
        if sheets is None:
            raise ValueError(f"{filename} contains no worksheets.")

        selected = None
        for sheet in sheets:
            if sheet.attrib.get("name") == sheet_name:
                selected = sheet
                break
        if selected is None:
            available = [sheet.attrib.get("name", "") for sheet in sheets]
            raise ValueError(
                f"Worksheet {sheet_name!r} not found in {filename}; available sheets: {available}"
            )

        relationship_id = selected.attrib[f"{{{relationship_ns}}}id"]
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relationship in rels_root.findall(f"{{{package_ns}}}Relationship"):
            if relationship.attrib.get("Id") == relationship_id:
                target = relationship.attrib.get("Target")
                break
        if target is None:
            raise ValueError(f"Worksheet relationship {relationship_id!r} is missing.")

        if target.startswith("/"):
            sheet_path = target.lstrip("/")
        else:
            sheet_path = str(PurePosixPath("xl") / target)

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f"{{{main_ns}}}si"):
                shared_strings.append(
                    "".join(node.text or "" for node in item.iter(f"{{{main_ns}}}t"))
                )

        worksheet_root = ET.fromstring(archive.read(sheet_path))
        sheet_data = worksheet_root.find(f"{{{main_ns}}}sheetData")
        if sheet_data is None:
            return []

        rows: list[list[object | None]] = []
        for row in sheet_data.findall(f"{{{main_ns}}}row"):
            values: dict[int, object | None] = {}
            for cell in row.findall(f"{{{main_ns}}}c"):
                reference = cell.attrib.get("r", "")
                index = _column_index(reference)
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{{{main_ns}}}v")

                if cell_type == "inlineStr":
                    inline = cell.find(f"{{{main_ns}}}is")
                    value: object | None = (
                        "".join(node.text or "" for node in inline.iter(f"{{{main_ns}}}t"))
                        if inline is not None
                        else ""
                    )
                elif value_node is None:
                    value = None
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text or "0")]
                elif cell_type in {"str", "e"}:
                    value = value_node.text or ""
                else:
                    raw = value_node.text or ""
                    try:
                        number = float(raw)
                        value = int(number) if number.is_integer() else number
                    except ValueError:
                        value = raw
                values[index] = value

            width = max(values, default=-1) + 1
            rows.append([values.get(index) for index in range(width)])
        return rows


def load_data(
    filename: Path, config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG
) -> list[TrainingExample]:
    """Load and validate the two-column Paper 3 training workbook."""
    rows = _xlsx_rows(filename)
    if not rows:
        raise ValueError(f"{filename} contains no data.")

    header = rows[0]
    expected = ["Generated String", "Production Rule Applications"]
    if header[:2] != expected:
        raise ValueError(f"{filename} must begin with columns {expected}.")

    examples: list[TrainingExample] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) < 2 or row[0] is None or row[1] is None:
            raise ValueError(f"Missing training value on row {row_number} of {filename}.")
        text = str(row[0]).strip()
        if not text or any(character not in config.terminals for character in text):
            raise ValueError(f"Invalid generated string on row {row_number}: {text!r}")
        try:
            applications = int(row[1]) #type: ignore
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid rule-application count on row {row_number}.") from error
        expected_applications = 2 * len(text) - 1
        if applications != expected_applications:
            raise ValueError(
                f"Row {row_number} has {applications} applications; a CNF derivation of "
                f"a length-{len(text)} string requires {expected_applications}."
            )
        if text in seen:
            raise ValueError(f"Duplicate training string on row {row_number}: {text!r}")
        seen.add(text)
        examples.append(TrainingExample(text, applications))

    if not examples:
        raise ValueError(f"{filename} contains no training examples.")
    return examples


@lru_cache(maxsize=None)
def all_possible_rules(
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> frozenset[Rule]:
    rules = {
        Rule(lhs, left, right)
        for lhs in config.nonterminals
        for left in config.nonterminals
        for right in config.nonterminals
    }
    rules.update(
        Rule(lhs, terminal)
        for lhs in config.nonterminals
        for terminal in config.terminals
    )
    return frozenset(rules)


ALL_RULES = all_possible_rules()


def productive_nonterminals(rules: Iterable[Rule]) -> set[str]:
    rules = tuple(rules)
    productive: set[str] = set()
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.lhs in productive:
                continue
            if rule.is_terminal or (
                rule.first in productive and rule.second in productive
            ):
                productive.add(rule.lhs)
                changed = True
    return productive


def reachable_nonterminals(
    rules: Iterable[Rule], config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG
) -> set[str]:
    rules = tuple(rules)
    reachable = {config.start_symbol}
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.lhs not in reachable or rule.is_terminal:
                continue
            for symbol in (rule.first, rule.second):
                if symbol not in reachable:
                    reachable.add(symbol)  # type: ignore[arg-type]
                    changed = True
    return reachable


def prune_useless_rules(
    rules: Iterable[Rule], config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG
) -> frozenset[Rule]:
    """Remove every rule that cannot participate in a start-symbol derivation."""
    unique = frozenset(rules)
    productive = productive_nonterminals(unique)
    productive_rules = frozenset(
        rule
        for rule in unique
        if rule.lhs in productive
        and (
            rule.is_terminal
            or (rule.first in productive and rule.second in productive)
        )
    )
    reachable = reachable_nonterminals(productive_rules, config)
    return frozenset(rule for rule in productive_rules if rule.lhs in reachable)


def is_valid_grammar(
    rules: Iterable[Rule],
    require_all_nonterminals: bool = False,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> bool:
    unique = frozenset(rules)
    if config.start_symbol not in productive_nonterminals(unique):
        return False
    if prune_useless_rules(unique, config) != unique:
        return False
    return not require_all_nonterminals or reachable_nonterminals(unique, config) == set(
        config.nonterminals
    )


def _productive_proof_rules(
    pool: frozenset[Rule],
    rng: random.Random,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> dict[str, Rule] | None:
    """Choose one acyclic productivity proof rule per nonterminal."""
    proven: dict[str, Rule] = {}
    while len(proven) < len(config.nonterminals):
        progress = False
        symbols = [symbol for symbol in config.nonterminals if symbol not in proven]
        rng.shuffle(symbols)
        for symbol in symbols:
            options = sorted(
                rule
                for rule in pool
                if rule.lhs == symbol
                and (
                    rule.is_terminal
                    or (rule.first in proven and rule.second in proven)
                )
            )
            if options:
                proven[symbol] = rng.choice(options)
                progress = True
        if not progress:
            return None
    return proven


def build_scaffold(
    pool: Iterable[Rule],
    rng: random.Random,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> frozenset[Rule] | None:
    """Select a useful, productive grammar reaching every configured symbol."""
    pool = frozenset(pool)
    if productive_nonterminals(pool) != set(config.nonterminals):
        return None

    proofs = _productive_proof_rules(pool, rng, config)
    if proofs is None:
        return None
    selected: set[Rule] = set(proofs.values())

    reached = {config.start_symbol}
    while reached != set(config.nonterminals):
        candidates = sorted(
            rule
            for rule in pool
            if not rule.is_terminal
            and rule.lhs in reached
            and (rule.first not in reached or rule.second not in reached)
        )
        if not candidates:
            return None
        rule = rng.choice(candidates)
        selected.add(rule)
        reached.update((rule.first, rule.second))  # type: ignore[arg-type]

    # The second scaffold pass adds one further rule per nonterminal.
    for symbol in config.nonterminals:
        extras = sorted(
            rule for rule in pool if rule.lhs == symbol and rule not in selected
        )
        if extras:
            selected.add(rng.choice(extras))

    scaffold = prune_useless_rules(selected, config)
    if not is_valid_grammar(scaffold, require_all_nonterminals=True, config=config):
        return None
    return scaffold


def random_individual(
    rng: random.Random, config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG
) -> Individual:
    scaffold = build_scaffold(all_possible_rules(config), rng, config)
    if scaffold is None:  # The complete search space always supports a scaffold.
        raise RuntimeError("Unable to construct an initial scaffolding grammar.")
    return Individual(scaffold)


def mutate(
    individual: Individual,
    rng: random.Random,
    maximum_attempts: int = 100,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> None:
    """Mutate genome symbols according to the original paper specification."""
    original = tuple(sorted(individual.rules))
    probability = 1.0 / (3 * len(original))

    for _ in range(maximum_attempts):
        changed = False
        mutated: list[Rule] = []
        for rule in original:
            lhs = rule.lhs
            first = rule.first
            second = rule.second

            if rng.random() < probability:
                lhs = rng.choice(
                    [symbol for symbol in config.nonterminals if symbol != lhs]
                )
                changed = True
            if rule.is_terminal:
                if rng.random() < probability:
                    first = rng.choice(
                        [terminal for terminal in config.terminals if terminal != first]
                    )
                    changed = True
            else:
                if rng.random() < probability:
                    first = rng.choice(
                        [symbol for symbol in config.nonterminals if symbol != first]
                    )
                    changed = True
                if rng.random() < probability:
                    second = rng.choice(
                        [symbol for symbol in config.nonterminals if symbol != second]
                    )
                    changed = True
            mutated.append(Rule(lhs, first, second))

        if not changed:
            return
        candidate = prune_useless_rules(mutated, config)
        if is_valid_grammar(candidate, config=config):
            if candidate != individual.rules:
                individual.rules = candidate
                individual.fitness = float("-inf")
            return


def crossover(
    first: Individual,
    second: Individual,
    rng: random.Random,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> tuple[Individual, Individual] | None:
    """Build two independently scaffolded offspring from both rule sets."""
    combined = first.rules | second.rules
    child_one = build_scaffold(combined, rng, config)
    child_two = build_scaffold(combined, rng, config)
    if child_one is None or child_two is None:
        return None
    return Individual(child_one), Individual(child_two)


def fitness_proportional_selection(
    population: Sequence[Individual],
    rng: random.Random,
) -> Individual:
    """Select from a random sample using fitness-proportional weights.

    Paper 2 specifies a fresh sample of ten for each parent selection.  Runs
    with a smaller test population use the complete population.  Clamping at
    zero is a defensive measure for the otherwise undefined case of a
    negative fitness weight; if every weight is zero, selection is uniform.
    """
    if not population:
        raise ValueError("Cannot select a parent from an empty population.")
    sample = rng.sample(
        population,
        k=min(SELECTION_SAMPLE_SIZE, len(population)),
    )
    weights = [max(0.0, individual.fitness) for individual in sample]
    if sum(weights) == 0.0:
        return rng.choice(sample)
    return rng.choices(sample, weights=weights, k=1)[0]


def accepts(
    rules: Iterable[Rule],
    text: str,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> bool:
    """Return whether a grammar derives text, using the CYK algorithm."""
    if not text:
        return False
    terminal_producers: dict[str, set[str]] = defaultdict(set)
    binary_producers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rule in rules:
        if rule.is_terminal:
            terminal_producers[rule.first].add(rule.lhs)
        else:
            binary_producers[(rule.first, rule.second)].add(rule.lhs)  # type: ignore[arg-type]

    length = len(text)
    table = [[set() for _ in range(length + 1)] for _ in range(length)]
    for index, terminal in enumerate(text):
        table[index][index + 1].update(terminal_producers[terminal])
    for span in range(2, length + 1):
        for start in range(length - span + 1):
            end = start + span
            cell = table[start][end]
            for split in range(start + 1, end):
                for left in table[start][split]:
                    for right in table[split][end]:
                        cell.update(binary_producers.get((left, right), ()))
    return config.start_symbol in table[0][length]


def collect_substrings(texts: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Collect each distinct training substring once for batched CYK parsing."""
    maximum = max(map(len, texts), default=0)
    groups: list[tuple[str, ...]] = [tuple()]
    for length in range(1, maximum + 1):
        values = {
            text[start : start + length]
            for text in texts
            for start in range(len(text) - length + 1)
        }
        groups.append(tuple(sorted(values)))
    return tuple(groups)


def accepts_many(
    rules: Iterable[Rule],
    texts: Sequence[str],
    substrings_by_length: Sequence[Sequence[str]] | None = None,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> list[bool]:
    """Run exact CYK membership while reusing shared substrings across examples."""
    if substrings_by_length is None:
        substrings_by_length = collect_substrings(texts)

    terminal_producers: dict[str, set[str]] = defaultdict(set)
    binary_producers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rule in rules:
        if rule.is_terminal:
            terminal_producers[rule.first].add(rule.lhs)
        else:
            binary_producers[(rule.first, rule.second)].add(rule.lhs)  # type: ignore[arg-type]

    derives: dict[str, set[str]] = {}
    if len(substrings_by_length) > 1:
        for terminal in substrings_by_length[1]:
            derives[terminal] = set(terminal_producers.get(terminal, ()))
    for length in range(2, len(substrings_by_length)):
        for fragment in substrings_by_length[length]:
            producers: set[str] = set()
            for split in range(1, length):
                for left in derives[fragment[:split]]:
                    for right in derives[fragment[split:]]:
                        producers.update(binary_producers.get((left, right), ()))
            derives[fragment] = producers
    return [config.start_symbol in derives.get(text, ()) for text in texts]


def bounded_language_profile(
    rules: Iterable[Rule],
    max_length: int,
    sample_limit: int,
    cell_limit: int = LANGUAGE_CELL_LIMIT,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> LanguageProfile:
    """Build a sample and distinct-language-size bound through ``max_length``.

    The distinct count is exact while every dynamic-programming cell remains
    below ``cell_limit``.  Once a cell saturates, its count becomes a lower
    bound.  Using that lower bound makes the resulting precision an upper
    bound, which avoids over-penalizing a grammar merely because enumeration
    was capped while still strongly penalizing very broad grammars.
    """
    if max_length < 1:
        raise ValueError("max_length must be at least 1.")
    if sample_limit < 1:
        raise ValueError("sample_limit must be at least 1.")
    rules = tuple(sorted(rules))
    cell_limit = max(cell_limit, sample_limit * 4)
    languages = {
        symbol: [set() for _ in range(max_length + 1)]
        for symbol in config.nonterminals
    }
    truncated = {
        symbol: [False for _ in range(max_length + 1)]
        for symbol in config.nonterminals
    }
    for rule in rules:
        if rule.is_terminal and max_length >= 1:
            languages[rule.lhs][1].add(rule.first)

    binary_rules = [rule for rule in rules if not rule.is_terminal]
    for length in range(2, max_length + 1):
        for rule in binary_rules:
            destination = languages[rule.lhs][length]
            for left_length in range(1, length):
                right_length = length - left_length
                left_strings = languages[rule.first][left_length]
                right_strings = languages[rule.second][right_length]  # type: ignore[index]
                if not left_strings or not right_strings:
                    continue
                if (
                    truncated[rule.first][left_length]
                    or truncated[rule.second][right_length]  # type: ignore[index]
                ):
                    truncated[rule.lhs][length] = True
                for left in sorted(left_strings):
                    for right in sorted(right_strings):
                        destination.add(left + right)
                        if len(destination) >= cell_limit:
                            truncated[rule.lhs][length] = True
                            break
                    if len(destination) >= cell_limit:
                        break
                if len(destination) >= cell_limit:
                    break

    candidates = [
        string
        for length in range(1, max_length + 1)
        for string in sorted(languages[config.start_symbol][length])
    ]
    language_size_lower_bound = sum(
        cell_limit
        if truncated[config.start_symbol][length]
        else len(languages[config.start_symbol][length])
        for length in range(1, max_length + 1)
    )
    exact_size = not any(truncated[config.start_symbol][1:])
    if len(candidates) <= sample_limit:
        return LanguageProfile(tuple(candidates), language_size_lower_bound, exact_size)

    signature = "|".join(str(rule) for rule in rules).encode("ascii")
    seed = int.from_bytes(hashlib.sha256(signature).digest()[:8], "big")
    rng = random.Random(seed)

    # Preserve coverage across lengths before filling the remaining slots.
    chosen: set[str] = set()
    populated_lengths = [
        length
        for length in range(1, max_length + 1)
        if languages[config.start_symbol][length]
    ]
    quota = max(1, sample_limit // max(1, len(populated_lengths)))
    for length in populated_lengths:
        values = sorted(languages[config.start_symbol][length])
        chosen.update(values if len(values) <= quota else rng.sample(values, quota))
    remaining = [value for value in candidates if value not in chosen]
    if len(chosen) < sample_limit:
        chosen.update(rng.sample(remaining, min(sample_limit - len(chosen), len(remaining))))
    elif len(chosen) > sample_limit:
        chosen = set(rng.sample(sorted(chosen), sample_limit))
    return LanguageProfile(
        tuple(sorted(chosen, key=lambda value: (len(value), value))),
        language_size_lower_bound,
        exact_size,
    )


def bounded_language_sample(
    rules: Iterable[Rule],
    max_length: int,
    sample_limit: int,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> tuple[str, ...]:
    """Return only the representative strings from a bounded language profile."""
    return bounded_language_profile(
        rules, max_length, sample_limit, config=config
    ).strings


def exact_bounded_language(
    rules: Iterable[Rule],
    max_length: int,
    storage_limit: int = TARGET_LANGUAGE_STORAGE_LIMIT,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> tuple[str, ...]:
    """Enumerate a grammar's exact distinct language through ``max_length``.

    This is reserved for target-equivalence verification, not ordinary fitness
    evaluation.  The storage limit bounds all dynamic-programming cells so a
    broad target grammar fails explicitly instead of exhausting memory.
    """
    if max_length < 1:
        raise ValueError("max_length must be at least 1.")
    if storage_limit < 1:
        raise ValueError("storage_limit must be at least 1.")

    ordered_rules = tuple(sorted(rules))
    languages = {
        symbol: [set() for _ in range(max_length + 1)]
        for symbol in config.nonterminals
    }
    stored_strings = 0

    for rule in ordered_rules:
        if not rule.is_terminal:
            continue
        destination = languages[rule.lhs][1]
        if rule.first not in destination:
            destination.add(rule.first)
            stored_strings += 1

    binary_rules = [rule for rule in ordered_rules if not rule.is_terminal]
    for length in range(2, max_length + 1):
        for rule in binary_rules:
            destination = languages[rule.lhs][length]
            for left_length in range(1, length):
                right_length = length - left_length
                left_strings = languages[rule.first][left_length]
                right_strings = languages[rule.second][right_length]  # type: ignore[index]
                for left in left_strings:
                    for right in right_strings:
                        value = left + right
                        if value in destination:
                            continue
                        destination.add(value)
                        stored_strings += 1
                        if stored_strings > storage_limit:
                            raise ValueError(
                                "Exact bounded-language enumeration exceeded the "
                                f"{storage_limit:,}-string safety limit. Reduce max_length "
                                "or use a narrower target grammar."
                            )

    return tuple(
        string
        for length in range(1, max_length + 1)
        for string in sorted(languages[config.start_symbol][length])
    )


def load_target_grammar(
    filename: Path,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
) -> TargetGrammar:
    """Load and validate the documented CNF target-grammar JSON format."""
    try:
        payload = json.loads(filename.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {filename} at line {error.lineno}, column {error.colno}."
        ) from error

    if not isinstance(payload, dict):
        raise ValueError("Target grammar JSON must contain one object.")
    required_fields = {"start_symbol", "max_length", "rules"}
    missing_fields = sorted(required_fields - payload.keys())
    if missing_fields:
        raise ValueError(
            "Target grammar JSON is missing: " + ", ".join(missing_fields) + "."
        )
    unknown_fields = sorted(payload.keys() - required_fields)
    if unknown_fields:
        raise ValueError(
            "Target grammar JSON has unknown fields: " + ", ".join(unknown_fields) + "."
        )
    if payload["start_symbol"] != START_SYMBOL:
        raise ValueError(f'Target grammar "start_symbol" must be "{START_SYMBOL}".')
    if config.start_symbol != START_SYMBOL:
        raise ValueError(f"The evolutionary start symbol must be {START_SYMBOL}.")

    max_length = payload["max_length"]
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise ValueError('Target grammar "max_length" must be an integer.')
    if not 1 <= max_length <= MAX_LANGUAGE_LENGTH:
        raise ValueError(
            f'Target grammar "max_length" must be between 1 and {MAX_LANGUAGE_LENGTH}.'
        )

    encoded_rules = payload["rules"]
    if not isinstance(encoded_rules, list) or not encoded_rules:
        raise ValueError('Target grammar "rules" must be a nonempty list.')
    parsed_rules: list[Rule] = []
    possible_rules = all_possible_rules(config)
    for index, encoded_rule in enumerate(encoded_rules):
        if not isinstance(encoded_rule, list) or len(encoded_rule) not in (2, 3):
            raise ValueError(
                f"Target rule {index} must be [lhs, terminal] or [lhs, left, right]."
            )
        if not all(isinstance(symbol, str) for symbol in encoded_rule):
            raise ValueError(f"Every symbol in target rule {index} must be a string.")
        rule = Rule(*encoded_rule)
        if rule not in possible_rules:
            raise ValueError(
                f"Target rule {index} ({rule}) is not a valid rule for the configured symbols."
            )
        parsed_rules.append(rule)

    rules = frozenset(parsed_rules)
    if len(rules) != len(parsed_rules):
        raise ValueError("Target grammar contains duplicate rules.")
    if not is_valid_grammar(rules, config=config):
        raise ValueError(
            "Target grammar must derive from N1 and contain no unreachable or "
            "unproductive rules."
        )

    language = exact_bounded_language(rules, max_length, config=config)
    if not language:
        raise ValueError(
            f"Target grammar derives no strings through length {max_length}."
        )
    return TargetGrammar(rules, max_length, language)


class TargetEquivalenceChecker:
    """Recognize equality with one target language through a finite length bound."""

    def __init__(
        self,
        target: TargetGrammar,
        config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
    ) -> None:
        self.target = target
        self.config = config
        self._target_substrings = collect_substrings(target.language)
        self._cache: dict[frozenset[Rule], bool] = {}

    def validate_examples(self, examples: Sequence[TrainingExample]) -> None:
        """Reject a target that cannot generate every positive training example."""
        texts = tuple(example.text for example in examples)
        accepted = accepts_many(
            self.target.rules,
            texts,
            collect_substrings(texts),
            self.config,
        )
        missing = [text for text, matched in zip(texts, accepted) if not matched]
        if missing:
            preview = ", ".join(repr(text) for text in missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(
                "The target grammar does not generate every positive training example; "
                f"missing {preview}{suffix}."
            )

    def find_equivalent(
        self,
        population: Sequence[Individual],
        evaluator: "FitnessEvaluator",
    ) -> Individual | None:
        """Return a bounded-language-equivalent candidate, if one is present."""
        target_size = len(self.target.language)
        checked_this_call: set[frozenset[Rule]] = set()
        for individual in population:
            rules = individual.rules
            if rules in checked_this_call:
                continue
            checked_this_call.add(rules)
            cached = self._cache.get(rules)
            if cached is not None:
                if cached:
                    return individual.copy()
                continue

            metrics = evaluator.breakdown(individual)
            if metrics.recall != 1.0:
                self._cache[rules] = False
                continue
            if self.target.max_length == evaluator.max_length:
                size = metrics.language_size_lower_bound
                exact_size = metrics.language_size_exact
            else:
                profile = bounded_language_profile(
                    rules,
                    self.target.max_length,
                    evaluator.language_sample_limit,
                    config=self.config,
                )
                size = profile.size_lower_bound
                exact_size = profile.exact_size

            equivalent = False
            if exact_size and size == target_size:
                inclusion = accepts_many(
                    rules,
                    self.target.language,
                    self._target_substrings,
                    self.config,
                )
                equivalent = all(inclusion)
            elif not exact_size and size <= target_size:
                # A highly ambiguous grammar can saturate an intermediate cell
                # even when its distinct start language is small.  This bounded
                # fallback prevents that truncation from creating a false negative.
                try:
                    candidate_language = exact_bounded_language(
                        rules,
                        self.target.max_length,
                        config=self.config,
                    )
                except ValueError:
                    candidate_language = tuple()
                equivalent = candidate_language == self.target.language

            self._cache[rules] = equivalent
            if equivalent:
                return individual.copy()
        return None


def levenshtein_distance(first: str, second: str) -> int:
    """Reference scalar Levenshtein distance used by tests and small inputs."""
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for first_index, first_character in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_character in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + (first_character != second_character),
                )
            )
        previous = current
    return previous[-1]


def minimum_levenshtein_distances(
    examples: Sequence[str], candidates: Sequence[str]
) -> list[int | None]:
    """Vectorize pairwise distances, excluding candidates longer than examples."""
    result: list[int | None] = [None] * len(examples)
    example_groups: dict[int, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        example_groups[len(example)].append(index)

    for example_length, indices in example_groups.items():
        allowed = [candidate for candidate in candidates if len(candidate) <= example_length]
        if not allowed:
            continue
        example_codes = np.array(
            [[ord(character) for character in examples[index]] for index in indices],
            dtype=np.int16,
        )
        candidate_lengths = np.array([len(candidate) for candidate in allowed], dtype=np.intp)
        candidate_codes = np.full(
            (len(allowed), example_length), -1, dtype=np.int16
        )
        for candidate_index, candidate in enumerate(allowed):
            candidate_codes[candidate_index, : len(candidate)] = [
                ord(character) for character in candidate
            ]

        pair_shape = (len(indices), len(allowed), example_length + 1)
        previous = np.broadcast_to(
            np.arange(example_length + 1, dtype=np.int16), pair_shape
        ).copy()
        for source_index in range(1, example_length + 1):
            current = np.empty_like(previous)
            current[:, :, 0] = source_index
            source_characters = example_codes[:, None, source_index - 1]
            for candidate_column in range(1, example_length + 1):
                substitution = (
                    source_characters != candidate_codes[None, :, candidate_column - 1]
                ).astype(np.int16)
                current[:, :, candidate_column] = np.minimum(
                    np.minimum(
                        previous[:, :, candidate_column] + 1,
                        current[:, :, candidate_column - 1] + 1,
                    ),
                    previous[:, :, candidate_column - 1] + substitution,
                )
            previous = current

        candidate_indices = np.arange(len(allowed), dtype=np.intp)
        distances = previous[:, candidate_indices, candidate_lengths]
        best = distances.min(axis=1)

        for local_index, original_index in enumerate(indices):
            result[original_index] = int(best[local_index])
    return result


class FitnessEvaluator:
    """Evaluate and cache grammars against the complete training workbook."""

    def __init__(
        self,
        examples: Sequence[TrainingExample],
        language_sample_limit: int = DEFAULT_LANGUAGE_SAMPLE_LIMIT,
        config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
    ) -> None:
        if language_sample_limit < 1:
            raise ValueError("language_sample_limit must be at least 1.")
        self.examples = tuple(examples)
        self.texts = tuple(example.text for example in self.examples)
        self.language_sample_limit = language_sample_limit
        self.config = config
        longest_example = max(len(example.text) for example in self.examples)
        if longest_example > MAX_LANGUAGE_LENGTH:
            raise ValueError(
                f"The specification bounds generated strings at length {MAX_LANGUAGE_LENGTH}, "
                f"but the data contains a length-{longest_example} example."
            )
        self.max_length = MAX_LANGUAGE_LENGTH
        self.substrings_by_length = collect_substrings(self.texts)
        self._cache: dict[frozenset[Rule], FitnessBreakdown] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def breakdown(self, individual: Individual) -> FitnessBreakdown:
        cached = self._cache.get(individual.rules)
        if cached is not None:
            individual.fitness = cached.fitness
            return cached

        exact = accepts_many(
            individual.rules,
            self.texts,
            self.substrings_by_length,
            self.config,
        )
        match_count = sum(exact)
        similarities = [1.0 if matched else 0.0 for matched in exact]
        language = bounded_language_profile(
            individual.rules,
            self.max_length,
            self.language_sample_limit,
            config=self.config,
        )

        unmatched_indices = [index for index, matched in enumerate(exact) if not matched]
        if unmatched_indices:
            unmatched_strings = [self.examples[index].text for index in unmatched_indices]
            distances = minimum_levenshtein_distances(unmatched_strings, language.strings)
            for index, distance in zip(unmatched_indices, distances):
                if distance is not None:
                    similarities[index] = 1.0 - distance / len(self.examples[index].text)

        example_count = len(self.examples)
        recall = match_count / example_count
        precision = (
            min(1.0, match_count / language.size_lower_bound)
            if language.size_lower_bound
            else 0.0
        )
        similarity = sum(similarities) / example_count
        value = (
            C1 * recall
            + C2 * precision
            + C3 * similarity
            - C4 * len(individual.rules)
        )
        result = FitnessBreakdown(
            fitness=value,
            recall=recall,
            precision=precision,
            similarity=similarity,
            matched_examples=match_count,
            language_size_lower_bound=language.size_lower_bound,
            language_size_exact=language.exact_size,
            rule_count=len(individual.rules),
        )
        individual.fitness = value
        self._cache[individual.rules] = result
        return result

    def score(self, individual: Individual) -> float:
        return self.breakdown(individual).fitness


def next_generation(
    population: Sequence[Individual],
    evaluator: FitnessEvaluator,
    rng: random.Random,
) -> list[Individual]:
    """Apply elitism, proportional selection, four-child crossover, and mutation."""
    population_size = len(population)
    elites = [
        individual.copy()
        for individual in sorted(population, key=lambda item: item.fitness, reverse=True)[
            : min(ELITE_COUNT, population_size)
        ]
    ]
    children: list[Individual] = []
    failed_crossovers = 0
    while len(elites) + len(children) < population_size:
        parent_one = fitness_proportional_selection(population, rng)
        parent_two = fitness_proportional_selection(population, rng)
        crossed = crossover(parent_one, parent_two, rng, evaluator.config)
        if crossed is None:
            failed_crossovers += 1
            if failed_crossovers < 1000:
                continue
            crossed = (
                random_individual(rng, evaluator.config),
                random_individual(rng, evaluator.config),
            )

        offspring = [parent_one.copy(), parent_two.copy(), crossed[0], crossed[1]]
        for child in offspring:
            mutate(child, rng, config=evaluator.config)
            evaluator.score(child)
        children.extend(offspring)

    return elites + children[: population_size - len(elites)]


def run(
    examples: Sequence[TrainingExample],
    generations: int,
    rng: random.Random,
    language_sample_limit: int = DEFAULT_LANGUAGE_SAMPLE_LIMIT,
    population_size: int = POPULATION_SIZE,
    show_progress: bool = True,
    config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG,
    target_equivalence_checker: TargetEquivalenceChecker | None = None,
    target_equivalence_interval: int = DEFAULT_TARGET_EQUIVALENCE_INTERVAL,
) -> RunResult:
    if generations < 1:
        raise ValueError("generations must be at least 1.")
    if population_size < ELITE_COUNT:
        raise ValueError(f"population_size must be at least {ELITE_COUNT}.")
    if target_equivalence_checker is not None and target_equivalence_interval < 1:
        raise ValueError("target equivalence interval must be at least 1.")
    if target_equivalence_checker is not None:
        target_equivalence_checker.validate_examples(examples)

    evaluator = FitnessEvaluator(examples, language_sample_limit, config)
    population = [random_individual(rng, config) for _ in range(population_size)]
    for individual in population:
        evaluator.score(individual)

    best = max(population, key=lambda item: item.fitness).copy()
    average_history = [sum(item.fitness for item in population) / population_size]
    best_history = [best.fitness]
    if show_progress:
        print(f"Generation 0: average = {average_history[-1]:.6f}, best = {best.fitness:.6f}")

    if target_equivalence_checker is not None:
        equivalent = target_equivalence_checker.find_equivalent(population, evaluator)
        if equivalent is not None:
            if show_progress:
                print(
                    "Bounded target equivalence found at generation 0 "
                    f"(through length {target_equivalence_checker.target.max_length})."
                )
            return RunResult(
                best,
                0,
                average_history,
                best_history,
                evaluator.breakdown(best),
                "target_equivalence",
                equivalent,
            )

    for generation in range(1, generations + 1):
        population = next_generation(
            population,
            evaluator,
            rng,
        )
        candidate = max(population, key=lambda item: item.fitness)
        if candidate.fitness > best.fitness:
            best = candidate.copy()
        average_history.append(sum(item.fitness for item in population) / population_size)
        best_history.append(candidate.fitness)
        if show_progress and (generation % 25 == 0 or generation == generations):
            print(
                f"Generation {generation}: average = {average_history[-1]:.6f}, "
                f"best = {candidate.fitness:.6f}"
            )

        should_check_target = (
            target_equivalence_checker is not None
            and (
                generation % target_equivalence_interval == 0
                or generation == generations
            )
        )
        if should_check_target:
            equivalent = target_equivalence_checker.find_equivalent(population, evaluator) #type: ignore
            if equivalent is not None:
                if show_progress:
                    print(
                        f"Bounded target equivalence found at generation {generation} "
                        f"(through length {target_equivalence_checker.target.max_length})." #type: ignore
                    )
                return RunResult(
                    best,
                    generation,
                    average_history,
                    best_history,
                    evaluator.breakdown(best),
                    "target_equivalence",
                    equivalent,
                )

    return RunResult(
        best,
        generations,
        average_history,
        best_history,
        evaluator.breakdown(best),
        "generation_limit",
        None,
    )


def save_fitness_graph(result: RunResult, output_file: Path) -> None:
    """Save Graph 1: average and best fitness for one algorithm run."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Matplotlib is required to create a fitness graph.") from error

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    generations = range(len(result.best_fitness))
    axis.plot(generations, result.average_fitness, label="Average fitness", linewidth=1.5)
    axis.plot(generations, result.best_fitness, label="Best fitness", linewidth=2)
    axis.set_title("Grammatical Evolution Fitness by Generation")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Fitness")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def save_trial_graph(
    results: Sequence[RunResult], seeds: Sequence[int], output_file: Path
) -> None:
    """Save Graph 2: best-fitness histories from five independent trials."""
    if len(results) != len(seeds):
        raise ValueError("Each trial result must have a corresponding seed.")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Matplotlib is required to create a fitness graph.") from error

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    for result, seed in zip(results, seeds):
        axis.plot(
            range(len(result.best_fitness)),
            result.best_fitness,
            label=f"Seed {seed}",
            linewidth=1.5,
        )
    axis.set_title("Best Grammatical-Evolution Fitness Across Five Trials")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Best fitness")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def format_grammar(
    rules: Iterable[Rule], config: GrammarConfig = DEFAULT_GRAMMAR_CONFIG
) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for rule in sorted(rules):
        rhs = rule.first if rule.is_terminal else f"{rule.first} {rule.second}"
        grouped[rule.lhs].append(rhs)
    return "\n".join(
        f"{symbol} -> {' | '.join(grouped[symbol])}"
        for symbol in config.nonterminals
        if grouped[symbol]
    )


def parse_arguments() -> argparse.Namespace:
    project_directory = Path(__file__).resolve().parents[2]
    data_directory = project_directory / "Data"
    if not data_directory.exists():
        data_directory = project_directory.parent / "Paper 1" / "Data"
    default_data = data_directory / "grammarData.xlsx"
    if not default_data.exists():
        default_data = data_directory / "Spreadsheets" / "E1Data.xlsx"
    parser = argparse.ArgumentParser(
        description="Run the sampled fitness-proportional-selection experiment."
    )
    parser.add_argument(
        "data",
        nargs="?",
        type=Path,
        default=default_data,
        help=(
            "training workbook (default: local Paper 2 Data, falling back to "
            "the corresponding Paper 1 Data directory)"
        ),
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
        help="number of generations (default: 1000)",
    )
    parser.add_argument("--seed", type=int, help="random seed for a repeatable run")
    parser.add_argument(
        "--nonterminals",
        type=int,
        default=DEFAULT_NONTERMINAL_COUNT,
        help="fixed number of nonterminals N1..N (minimum 2; default: 4)",
    )
    parser.add_argument(
        "--language-sample-limit",
        type=int,
        default=DEFAULT_LANGUAGE_SAMPLE_LIMIT,
        help="representative language size for Levenshtein smoothing (default: 256)",
    )
    parser.add_argument(
        "--target-grammar",
        type=Path,
        help=(
            "target CNF grammar JSON; stop when a candidate has the same distinct "
            "language through the JSON max_length"
        ),
    )
    parser.add_argument(
        "--target-equivalence-interval",
        type=int,
        default=DEFAULT_TARGET_EQUIVALENCE_INTERVAL,
        help="generations between target-equivalence checks (default: 25)",
    )
    graph_mode = parser.add_mutually_exclusive_group()
    graph_mode.add_argument(
        "--graph",
        action="store_true",
        help="save one run's average/best fitness graph",
    )
    graph_mode.add_argument(
        "--graphs",
        action="store_true",
        help="run five seeded trials and save both Paper 2-style graphs",
    )
    parser.add_argument(
        "--graph-file",
        type=Path,
        default=project_directory / "Graphs" / "fitness_by_generation.png",
        help="graph path (default: ../Graphs/fitness_by_generation.png)",
    )
    parser.add_argument(
        "--graph-directory",
        type=Path,
        default=project_directory / "Graphs",
        help="directory for --graphs outputs (default: ../Graphs)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress generation progress")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    try:
        config = GrammarConfig.from_count(args.nonterminals)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    examples = load_data(args.data, config)
    target_checker: TargetEquivalenceChecker | None = None
    if args.target_grammar is not None:
        try:
            target = load_target_grammar(args.target_grammar, config)
            target_checker = TargetEquivalenceChecker(target, config)
            target_checker.validate_examples(examples)
        except (OSError, ValueError) as error:
            raise SystemExit(f"error: {error}") from error
        if not args.quiet:
            print(
                f"Loaded target grammar with {len(target.rules)} rules and "
                f"{len(target.language)} distinct strings through length "
                f"{target.max_length}."
            )
    if args.graphs:
        first_seed = args.seed if args.seed is not None else 0
        seeds = [first_seed + offset for offset in range(TRIAL_COUNT)]
        results: list[RunResult] = []
        for index, seed in enumerate(seeds):
            print(f"\n=== Trial {index + 1}/{TRIAL_COUNT} (seed {seed}) ===")
            results.append(
                run(
                    examples,
                    args.generations,
                    random.Random(seed),
                    language_sample_limit=args.language_sample_limit,
                    show_progress=not args.quiet,
                    config=config,
                    target_equivalence_checker=target_checker,
                    target_equivalence_interval=args.target_equivalence_interval,
                )
            )

        print("\nFive-trial summary:")
        for index, (seed, trial_result) in enumerate(zip(seeds, results), start=1):
            metrics = trial_result.best_breakdown
            if metrics is None:
                metric_text = ""
            else:
                precision_label = (
                    "precision" if metrics.language_size_exact else "precision_upper_bound"
                )
                metric_text = (
                    f", recall={metrics.recall:.6f}, "
                    f"{precision_label}={metrics.precision:.6f}, "
                    f"rules={metrics.rule_count}"
                )
            stop_text = (
                f", stopped={trial_result.termination_reason}"
                f"@{trial_result.completed_generations}"
            )
            print(
                f"Trial {index} (seed {seed}): "
                f"best={trial_result.best.fitness:.12g}{metric_text}{stop_text}"
            )
        single_graph = args.graph_directory / "fitness_average_and_best.png"
        trial_graph = args.graph_directory / "fitness_five_trials.png"
        save_fitness_graph(results[0], single_graph)
        save_trial_graph(results, seeds, trial_graph)
        print(f"\nSaved Graph 1: {single_graph.resolve()}")
        print(f"Saved Graph 2: {trial_graph.resolve()}")
        result = results[0]
    else:
        result = run(
            examples,
            args.generations,
            random.Random(args.seed),
            language_sample_limit=args.language_sample_limit,
            show_progress=not args.quiet,
            config=config,
            target_equivalence_checker=target_checker,
            target_equivalence_interval=args.target_equivalence_interval,
        )

    if args.graph:
        save_fitness_graph(result, args.graph_file)
        print(f"Fitness graph: {args.graph_file.resolve()}")

    print(f"\nFinished after {result.completed_generations} generation(s).")
    if result.termination_reason == "target_equivalence" and target_checker is not None:
        print(
            "Termination: bounded target-language equivalence found through length "
            f"{target_checker.target.max_length}."
        )
    else:
        print("Termination: generation limit reached.")
    print(f"Best fitness: {result.best.fitness:.12g}")
    if result.best_breakdown is not None:
        metrics = result.best_breakdown
        language_label = (
            str(metrics.language_size_lower_bound)
            if metrics.language_size_exact
            else f">= {metrics.language_size_lower_bound}"
        )
        print(
            f"Recall: {metrics.recall:.6f} "
            f"({metrics.matched_examples}/{len(examples)} examples)"
        )
        precision_label = "Precision" if metrics.language_size_exact else "Precision (upper bound)"
        print(f"{precision_label}: {metrics.precision:.6f}")
        print(f"Average similarity: {metrics.similarity:.6f}")
        print(f"Generated strings through length {MAX_LANGUAGE_LENGTH}: {language_label}")
    print(f"Production rules: {len(result.best.rules)}")
    print(f"Nonterminals: {', '.join(config.nonterminals)}")
    if result.target_equivalent is not None and result.target_equivalent.rules == result.best.rules:
        print("Grammar (fitness-best and target-equivalent):")
    else:
        print("Fitness-best grammar:")
    print(format_grammar(result.best.rules, config))
    if (
        result.target_equivalent is not None
        and result.target_equivalent.rules != result.best.rules
    ):
        print(
            "Target-equivalent grammar "
            f"(fitness {result.target_equivalent.fitness:.12g}):"
        )
        print(format_grammar(result.target_equivalent.rules, config))


if __name__ == "__main__":
    main()
