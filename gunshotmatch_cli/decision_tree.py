#!/usr/bin/env python3
#
#  decision_tree.py
"""
Decision tree utilities.
"""
#
#  Copyright © 2020-2023 Dominic Davis-Foster <dominic@davis-foster.co.uk>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.
#

# stdlib
from typing import List, Tuple

# 3rd party
from gunshotmatch_pipeline.decision_tree import (
		data_from_projects,
		fit_decision_tree,
		get_feature_names,
		visualise_decision_tree
		)
from gunshotmatch_pipeline.projects import Projects
from gunshotmatch_pipeline.utils import project_plural
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import]

__all__ = ("train_decision_tree", )


def train_decision_tree(
		projects: Projects,
		*,
		random_forest: bool = True,
		) -> Tuple[RandomForestClassifier, List[str], List[str]]:
	"""
	Train a decision tree on the given projects.

	:param projects:
	:param random_forest: Use a random forest classifier.

	.. versionchanged:: 0.4.0  Added ``random_forest`` keyword argument. Behaviour is unchanged with default value.
	"""

	print(f"Training decision tree on {len(projects)} {project_plural(len(projects))}:")
	for project in projects.per_project_settings:
		print(f"  {project}")

	data, factorize_map = data_from_projects(projects)
	data.to_csv("decision_tree_df.csv")

	# TODO: cache loaded data prior to training, to save time next time round

	# Fit the classifier with default hyper-parameters
	# clf = sklearn.tree.DecisionTreeClassifier(random_state=20230703)
	# clf = sklearn.tree.DecisionTreeClassifier(random_state=20231020)
	# fit_decision_tree(data, clf)
	# visualise_decision_tree(data, clf, factorize_map, filename="trees/decision_tree")
	# visualise_decision_tree(data, clf, factorize_map, filename="trees/decision_tree", filetype="png")

	if random_forest:
		clf = RandomForestClassifier(n_jobs=4, random_state=20231020)
	else:
		clf = DecisionTreeClassifier(random_state=20231020)
	fit_decision_tree(data, clf)
	visualise_decision_tree(data, clf, factorize_map, filename="trees/decision_tree")
	return clf, factorize_map, get_feature_names(data)
