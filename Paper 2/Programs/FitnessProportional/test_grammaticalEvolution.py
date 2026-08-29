from __future__ import annotations

import math
import json
import random
import tempfile
import unittest
from pathlib import Path

import FitnessProportional.FitnessProportional as ge


DATA_FILE = (
    Path(__file__).resolve().parents[3]
    / "Paper 1"
    / "Data"
    / "Spreadsheets"
    / "E1Data.xlsx"
)


class GrammaticalEvolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = ge.load_data(DATA_FILE)
        cls.target_rules = frozenset(
            {
                ge.Rule("N1", "N2", "N3"),
                ge.Rule("N1", "N2", "N4"),
                ge.Rule("N2", "a"),
                ge.Rule("N3", "N1", "N4"),
                ge.Rule("N4", "b"),
                ge.Rule("N4", "c"),
            }
        )
        cls.universal_rules = frozenset(
            {
                ge.Rule("N1", "N1", "N1"),
                ge.Rule("N1", "a"),
                ge.Rule("N1", "b"),
                ge.Rule("N1", "c"),
            }
        )

    def test_workbook_has_the_specified_training_data(self) -> None:
        self.assertEqual(len(self.examples), 250)
        self.assertEqual(self.examples[0], ge.TrainingExample("ab", 3))
        self.assertTrue(
            all(example.applications == 2 * len(example.text) - 1 for example in self.examples)
        )

    def test_target_equivalent_grammar_matches_every_example(self) -> None:
        self.assertTrue(ge.is_valid_grammar(self.target_rules))
        self.assertTrue(all(ge.accepts(self.target_rules, item.text) for item in self.examples))
        self.assertEqual(
            ge.accepts_many(self.target_rules, [item.text for item in self.examples]),
            [True] * len(self.examples),
        )
        evaluator = ge.FitnessEvaluator(self.examples, language_sample_limit=64)
        individual = ge.Individual(self.target_rules)
        breakdown = evaluator.breakdown(individual)
        self.assertAlmostEqual(breakdown.recall, 1.0)
        self.assertAlmostEqual(
            breakdown.precision,
            250 / breakdown.language_size_lower_bound,
        )
        self.assertAlmostEqual(breakdown.similarity, 1.0)
        self.assertAlmostEqual(
            breakdown.fitness,
            ge.C1 + ge.C2 * breakdown.precision + ge.C3 - ge.C4 * 6,
        )
        self.assertAlmostEqual(
            breakdown.fitness, breakdown.core_fitness  # type: ignore[arg-type]
        )

    def test_revised_fitness_penalizes_overgeneralization(self) -> None:
        evaluator = ge.FitnessEvaluator(self.examples, language_sample_limit=64)
        universal = ge.Individual(self.universal_rules)
        target = ge.Individual(self.target_rules)
        self.assertTrue(all(ge.accepts(self.universal_rules, item.text) for item in self.examples))
        universal_metrics = evaluator.breakdown(universal)
        target_metrics = evaluator.breakdown(target)
        self.assertAlmostEqual(universal_metrics.recall, 1.0)
        self.assertLess(universal_metrics.precision, 0.1)
        self.assertGreater(target_metrics.fitness, universal_metrics.fitness)

    def test_fitness_plateau_transform_has_inclusive_boundaries(self) -> None:
        self.assertEqual(ge.flatten_fitness(94.999), 94.999)
        self.assertEqual(ge.flatten_fitness(95.0), 95.0)
        self.assertEqual(ge.flatten_fitness(102.5), 95.0)
        self.assertEqual(ge.flatten_fitness(110.0), 95.0)
        self.assertEqual(ge.flatten_fitness(110.001), 110.001)

    def test_overgeneralized_fitness_is_flattened_but_core_is_retained(self) -> None:
        evaluator = ge.FitnessEvaluator(self.examples, language_sample_limit=64)
        metrics = evaluator.breakdown(ge.Individual(self.universal_rules))

        self.assertIsNotNone(metrics.core_fitness)
        assert metrics.core_fitness is not None
        self.assertGreaterEqual(metrics.core_fitness, ge.FLATTENING_LOWER_BOUND)
        self.assertLessEqual(metrics.core_fitness, ge.FLATTENING_UPPER_BOUND)
        self.assertEqual(metrics.fitness, ge.FLATTENED_FITNESS)

    def test_selection_uses_transformed_fitness_within_a_sample_of_ten(self) -> None:
        class RecordingRandom:
            def __init__(self) -> None:
                self.sample_size: int | None = None
                self.weights: list[float] | None = None

            def sample(self, population, k):  # type: ignore[no-untyped-def]
                self.sample_size = k
                return list(population)[:k]

            def choices(self, population, weights, k):  # type: ignore[no-untyped-def]
                self.weights = list(weights)
                return [population[-1]]

            def choice(self, population):  # type: ignore[no-untyped-def]
                return population[0]

        population = [
            ge.Individual(self.target_rules, float(index))
            for index in range(1, 13)
        ]
        rng = RecordingRandom()

        selected = ge.fitness_proportional_selection(
            population, rng  # type: ignore[arg-type]
        )

        self.assertEqual(rng.sample_size, 10)
        self.assertEqual(rng.weights, [float(index) for index in range(1, 11)])
        self.assertIs(selected, population[9])

    def test_scaffolds_are_reachable_productive_and_pruned(self) -> None:
        for seed in range(20):
            individual = ge.random_individual(random.Random(seed))
            self.assertTrue(ge.is_valid_grammar(individual.rules, require_all_nonterminals=True))
            self.assertEqual(ge.prune_useless_rules(individual.rules), individual.rules)

    def test_mutation_and_crossover_preserve_validity(self) -> None:
        rng = random.Random(101)
        first = ge.random_individual(rng)
        second = ge.random_individual(rng)
        crossed = ge.crossover(first, second, rng)
        self.assertIsNotNone(crossed)
        assert crossed is not None
        for child in crossed:
            self.assertTrue(ge.is_valid_grammar(child.rules, require_all_nonterminals=True))
            for _ in range(10):
                ge.mutate(child, rng)
                self.assertTrue(ge.is_valid_grammar(child.rules))

    def test_mutation_does_not_add_production_rules(self) -> None:
        for seed in range(20):
            individual = ge.random_individual(random.Random(seed))
            original_rule_count = len(individual.rules)

            ge.mutate(individual, random.Random(seed + 1000))

            self.assertLessEqual(len(individual.rules), original_rule_count)
            self.assertTrue(ge.is_valid_grammar(individual.rules))

    def test_configurable_nonterminal_count_is_used_everywhere(self) -> None:
        config = ge.GrammarConfig.from_count(6)
        self.assertEqual(
            config.nonterminals,
            ("N1", "N2", "N3", "N4", "N5", "N6"),
        )
        self.assertEqual(len(ge.all_possible_rules(config)), 6**3 + 6 * 3)

        rng = random.Random(606)
        first = ge.random_individual(rng, config)
        second = ge.random_individual(rng, config)
        for individual in (first, second):
            self.assertTrue(
                ge.is_valid_grammar(
                    individual.rules,
                    require_all_nonterminals=True,
                    config=config,
                )
            )

        crossed = ge.crossover(first, second, rng, config)
        self.assertIsNotNone(crossed)
        assert crossed is not None
        for child in crossed:
            ge.mutate(child, rng, config=config)
            self.assertTrue(ge.is_valid_grammar(child.rules, config=config))
            self.assertTrue(
                set(rule.lhs for rule in child.rules) <= set(config.nonterminals)
            )

        evaluator = ge.FitnessEvaluator(
            self.examples[:8], language_sample_limit=16, config=config
        )
        self.assertTrue(math.isfinite(evaluator.score(crossed[0])))
        formatted = ge.format_grammar(first.rules, config)
        self.assertIn("N5 ->", formatted)
        self.assertIn("N6 ->", formatted)

        with self.assertRaises(ValueError):
            ge.GrammarConfig.from_count(1)

    def test_bounded_language_generation(self) -> None:
        profile = ge.bounded_language_profile(self.target_rules, 6, 100)
        language = set(profile.strings)
        self.assertTrue(profile.exact_size)
        self.assertEqual(profile.size_lower_bound, 14)
        self.assertEqual(
            language,
            {
                "ab",
                "ac",
                "aabb",
                "aabc",
                "aacb",
                "aacc",
                "aaabbb",
                "aaabbc",
                "aaabcb",
                "aaabcc",
                "aaacbb",
                "aaacbc",
                "aaaccb",
                "aaaccc",
            },
        )

    def test_target_grammar_json_loads_and_requires_n1(self) -> None:
        encoded_rules = [
            [rule.lhs, rule.first]
            if rule.is_terminal
            else [rule.lhs, rule.first, rule.second]
            for rule in sorted(self.target_rules)
        ]
        payload = {
            "start_symbol": "N1",
            "max_length": 6,
            "rules": encoded_rules,
        }
        with tempfile.TemporaryDirectory() as directory:
            target_file = Path(directory) / "target.json"
            target_file.write_text(json.dumps(payload), encoding="utf-8")
            target = ge.load_target_grammar(target_file)

            self.assertEqual(target.rules, self.target_rules)
            self.assertEqual(target.max_length, 6)
            self.assertEqual(len(target.language), 14)

            payload["start_symbol"] = "N2"
            target_file.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be.*N1"):
                ge.load_target_grammar(target_file)

    def test_bounded_target_equivalence_rejects_overgeneralization(self) -> None:
        target = ge.TargetGrammar(
            self.target_rules,
            6,
            ge.exact_bounded_language(self.target_rules, 6),
        )
        checker = ge.TargetEquivalenceChecker(target)
        checker.validate_examples(self.examples)
        evaluator = ge.FitnessEvaluator(self.examples, language_sample_limit=64)
        universal = ge.Individual(self.universal_rules)
        equivalent = ge.Individual(self.target_rules)

        match = checker.find_equivalent([universal, equivalent], evaluator)

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.rules, self.target_rules)
        self.assertIsNone(checker.find_equivalent([universal], evaluator))

    def test_target_checker_rejects_an_inconsistent_positive_dataset(self) -> None:
        target = ge.TargetGrammar(
            self.target_rules,
            6,
            ge.exact_bounded_language(self.target_rules, 6),
        )
        checker = ge.TargetEquivalenceChecker(target)
        with self.assertRaisesRegex(ValueError, "does not generate every positive"):
            checker.validate_examples([ge.TrainingExample("a", 1)])

    def test_run_checks_target_on_the_configured_cadence(self) -> None:
        class SecondCheckMatches:
            def __init__(self) -> None:
                self.calls = 0

            def validate_examples(self, examples):  # type: ignore[no-untyped-def]
                return None

            def find_equivalent(self, population, evaluator):  # type: ignore[no-untyped-def]
                self.calls += 1
                return population[0].copy() if self.calls == 2 else None

        checker = SecondCheckMatches()
        result = ge.run(
            self.examples[:2],
            generations=5,
            rng=random.Random(91),
            language_sample_limit=16,
            population_size=4,
            show_progress=False,
            target_equivalence_checker=checker,  # type: ignore[arg-type]
            target_equivalence_interval=2,
        )

        self.assertEqual(checker.calls, 2)
        self.assertEqual(result.completed_generations, 2)
        self.assertEqual(result.termination_reason, "target_equivalence")
        self.assertIsNotNone(result.target_equivalent)
        self.assertEqual(len(result.best_fitness), 3)

    def test_target_equivalence_interval_is_validated(self) -> None:
        target = ge.TargetGrammar(
            self.target_rules,
            6,
            ge.exact_bounded_language(self.target_rules, 6),
        )
        with self.assertRaisesRegex(ValueError, "interval must be at least 1"):
            ge.run(
                self.examples[:2],
                generations=1,
                rng=random.Random(0),
                population_size=4,
                show_progress=False,
                target_equivalence_checker=ge.TargetEquivalenceChecker(target),
                target_equivalence_interval=0,
            )

    def test_vectorized_levenshtein_matches_reference(self) -> None:
        examples = ["ab", "aabb", "aaaccc", "abc"]
        candidates = ["a", "ac", "aabc", "aaabbb", "bc"]
        actual = ge.minimum_levenshtein_distances(examples, candidates)
        expected = [
            min(
                ge.levenshtein_distance(example, candidate)
                for candidate in candidates
                if len(candidate) <= len(example)
            )
            for example in examples
        ]
        self.assertEqual(actual, expected)

    def test_one_small_generation_is_finite_and_valid(self) -> None:
        subset = self.examples[:20]
        result = ge.run(
            subset,
            generations=1,
            rng=random.Random(7),
            language_sample_limit=32,
            population_size=8,
            show_progress=False,
        )
        self.assertEqual(result.completed_generations, 1)
        self.assertEqual(len(result.average_fitness), 2)
        self.assertTrue(math.isfinite(result.best.fitness))
        self.assertTrue(ge.is_valid_grammar(result.best.rules))
        self.assertIsNotNone(result.best_breakdown)
        self.assertTrue(
            all(
                earlier <= later
                for earlier, later in zip(result.best_fitness, result.best_fitness[1:])
            )
        )

    def test_both_graphs_are_valid_png_files(self) -> None:
        trial_results = [
            ge.RunResult(
                ge.Individual(self.target_rules, 101.4),
                2,
                [float(seed), float(seed + 1), float(seed + 2)],
                [99.0 + seed, 100.0 + seed, 101.0 + seed],
            )
            for seed in range(5)
        ]
        with tempfile.TemporaryDirectory() as directory:
            single_file = Path(directory) / "single.png"
            trials_file = Path(directory) / "trials.png"
            ge.save_fitness_graph(trial_results[0], single_file)
            ge.save_trial_graph(trial_results, list(range(5)), trials_file)
            self.assertEqual(single_file.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(trials_file.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
