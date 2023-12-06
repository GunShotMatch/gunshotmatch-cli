#!/usr/bin/env python3
#
#  decision_tree.py
"""
Create decision tree and predict class of an unknown sample.
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

# 3rd party
import click

__all__ = ("decision_tree", )


@click.argument("projects_toml", default="projects.toml")
@click.argument("unknown_toml", default="unknown.toml")
@click.command()
def decision_tree(projects_toml: str = "projects.toml", unknown_toml: str = "unknown.toml") -> None:
	"""
	Create decision tree and predict class of an unknown sample.
	"""

	# 3rd party
	import numpy
	import sklearn.tree  # type: ignore[import]
	from domdf_python_tools.paths import PathPlus
	from gunshotmatch_pipeline.decision_tree import (
			data_from_projects,
			data_from_unknown,
			fit_decision_tree,
			get_feature_names,
			visualise_decision_tree
			)
	from gunshotmatch_pipeline.projects import Projects
	from gunshotmatch_pipeline.unknowns import UnknownSettings
	from gunshotmatch_pipeline.utils import project_plural
	from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]

	# root_dir = PathPlus(__file__).parent.abspath()
	# trees_dir = root_dir / "trees"
	# decision_tree_data_dir = root_dir / "decision_tree_data"
	# decision_tree_data_dir.maybe_make()

	projects = Projects.from_toml(PathPlus("projects.toml").read_text())
	print(f"Training decision tree on {len(projects)} {project_plural(len(projects))}:")
	for project in projects.per_project_settings:
		print(f"  {project}")

	data, factorize_map = data_from_projects(projects)
	data.to_csv("decision_tree_df.csv")

	# TODO: cache loaded data prior to training, to save time next time round

	# Fit the classifier with default hyper-parameters
	# clf = sklearn.tree.DecisionTreeClassifier(random_state=20230703)
	clf = sklearn.tree.DecisionTreeClassifier(random_state=20231020)
	fit_decision_tree(data, clf)
	visualise_decision_tree(data, clf, factorize_map, filename="trees/decision_tree")
	visualise_decision_tree(data, clf, factorize_map, filename="trees/decision_tree", filetype="png")

	forest_clf = RandomForestClassifier(n_jobs=4, random_state=20231020)
	fit_decision_tree(data, forest_clf)
	visualise_decision_tree(data, forest_clf, factorize_map, filename="trees/decision_tree")

	unknowns = UnknownSettings.from_toml(PathPlus("unknown-example.toml").read_text())
	# print(data_from_unknown(unknowns))

	unknown_sample = data_from_unknown(unknowns, feature_names=get_feature_names(data))
	predicted_class = forest_clf.predict(unknown_sample)
	print(predicted_class)
	print(factorize_map[predicted_class[0]])

	classifier = forest_clf
	proba = classifier.predict_proba(unknown_sample)
	print("proba:", proba.tolist())
	argmax = numpy.argmax(proba, axis=1)
	argsort = numpy.argsort(proba, axis=1)
	# print("argmax:", argmax.tolist())
	# print("argsort:", list(reversed(argsort.tolist()[0])))
	# print("classifier.classes_:", list(classifier.classes_))
	# print("Ranked:", [factorize_map[cls] for cls in reversed(argsort.tolist()[0])])
	# print("       ", sorted(classifier.predict_proba(unknown_sample)[0], reverse=True))

	for idx, (propellant, probability) in enumerate(
			zip([factorize_map[cls] for cls in reversed(argsort.tolist()[0])],
				sorted(classifier.predict_proba(unknown_sample)[0], reverse=True))
			):
		if probability:
			print(idx + 1, probability, propellant)

	# take = classifier.classes_.take(argmax, axis=0)
	# print("take:", list(take))


if __name__ == "__main__":
	decision_tree()
