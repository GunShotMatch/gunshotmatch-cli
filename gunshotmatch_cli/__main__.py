#!/usr/bin/env python3
#
#  __main__.py
"""
GunShotMatch Command-Line Interface.
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
from consolekit.options import version_option

# this package
from gunshotmatch_cli.versions import version_callback

__all__ = ("decision_tree", "gunshotmatch", "projects", "unknown")


@version_option(version_callback)
@click.group()
def main() -> None:
	"""
	GunShotMatch command-line interface.
	"""  # noqa: D403  (false positive)


@click.argument("projects_toml", default="projects.toml")
@main.command()
def projects(projects_toml: str = "projects.toml") -> None:
	"""
	Pipeline for creating projects from raw datafiles.
	"""

	# 3rd party
	from domdf_python_tools.paths import PathPlus
	from gunshotmatch_pipeline.exporters import verify_saved_project, write_combined_csv, write_matches_json
	from gunshotmatch_pipeline.projects import Projects, process_projects
	from gunshotmatch_pipeline.utils import project_plural
	from libgunshotmatch.peak import write_alignment
	from libgunshotmatch.project import Project

	projects = Projects.from_toml(PathPlus(projects_toml).read_text())
	output_dir = PathPlus(projects.global_settings.output_directory).abspath()

	print(f"Processing {len(projects)} {project_plural(len(projects))}:")
	for project_name in projects.per_project_settings:
		print(f"  {project_name}")
	print(f"Saving to {output_dir.as_posix()!r}")

	for project in process_projects(projects, output_dir, recreate=False):
		project_from_disk = Project.from_file(output_dir / (project.name + ".gsmp"))
		verify_saved_project(project, project_from_disk)

		# for repeat in project.datafile_data.values():
		# 	datafile = repeat.datafile

		# 	verify_saved_file = False
		# 	if verify_saved_file:
		# 		from_file = Datafile.from_file(datafile_export_filename)
		# 		verify_saved_datafile(datafile, from_file)

		write_alignment(project.alignment, project.name, output_dir)
		for repeat in project.datafile_data.values():
			write_combined_csv(repeat, output_dir)

		# Matches Sheet
		# MatchesCSVExporter(
		# 		os.path.join(output_dir, project.name + "_MATCHES.csv"), project, minutes=True, n_hits=5
		# 		)

		write_matches_json(project, output_dir)

		assert project.consolidated_peaks is not None
		print(project.consolidated_peaks)
		print(len(project.consolidated_peaks))


@click.argument("unknown_toml", default="unknown.toml")
@main.command()
def unknown(unknown_toml: str = "unknown.toml") -> None:
	"""
	Pipeline for unknown propellant/OGSR sample.
	"""

	# 3rd party
	from domdf_python_tools.paths import PathPlus
	from gunshotmatch_pipeline.unknowns import UnknownSettings, process_unknown

	# from gunshotmatch_pipeline.exporters import write_matches_json

	unknown = UnknownSettings.from_toml(PathPlus(unknown_toml).read_text())
	# print(unknown)
	# print(unknown.load_method())
	# print(unknown.load_config())
	print("Processing unknown", unknown.name)

	project = process_unknown(unknown, unknown.output_directory, recreate=False)

	# write_matches_json(project, PathPlus(unknown.output_directory))
	assert project.consolidated_peaks is not None
	print(len(project.consolidated_peaks))

	# print(ms_comparison_df)


@click.option("-p", "--projects", "projects_toml", default="projects.toml")
@click.option("-o", "--unknown", "unknown_toml", default="unknown.toml")
@main.command()
def decision_tree(projects_toml: str = "projects.toml", unknown_toml: str = "unknown.toml") -> None:
	"""
	Create decision tree and predict class of an unknown sample.
	"""

	# 3rd party
	import numpy
	from domdf_python_tools.paths import PathPlus
	from gunshotmatch_pipeline.decision_tree import data_from_unknown
	from gunshotmatch_pipeline.projects import Projects
	from gunshotmatch_pipeline.unknowns import UnknownSettings

	# this package
	from gunshotmatch_cli.decision_tree import train_decision_tree

	# root_dir = PathPlus(__file__).parent.abspath()
	# trees_dir = root_dir / "trees"
	# decision_tree_data_dir = root_dir / "decision_tree_data"
	# decision_tree_data_dir.maybe_make()

	projects = Projects.from_toml(PathPlus(projects_toml).read_text())
	unknown = UnknownSettings.from_toml(PathPlus(unknown_toml).read_text())

	classifier, factorize_map, feature_names = train_decision_tree(projects)

	print("\nPreicting class for unknown", unknown.name)

	unknown_sample = data_from_unknown(unknown, feature_names=feature_names)
	# predicted_class = classifier.predict(unknown_sample)
	# print(predicted_class)
	# print(factorize_map[predicted_class[0]])

	proba = classifier.predict_proba(unknown_sample)
	# print("proba:", proba.tolist())
	# argmax = numpy.argmax(proba, axis=1)
	argsort = numpy.argsort(proba, axis=1)
	# print("argmax:", argmax.tolist())
	# print("argsort:", list(reversed(argsort.tolist()[0])))
	# print("classifier.classes_:", list(classifier.classes_))
	# print("Ranked:", [factorize_map[cls] for cls in reversed(argsort.tolist()[0])])
	# print("       ", sorted(classifier.predict_proba(unknown_sample)[0], reverse=True))

	class_names = [factorize_map[cls] for cls in reversed(argsort.tolist()[0])]
	probabilities = sorted(classifier.predict_proba(unknown_sample)[0], reverse=True)
	for idx, (propellant, probability) in enumerate(zip(class_names, probabilities)):
		if probability:
			print(idx + 1, probability, propellant)

	# take = classifier.classes_.take(argmax, axis=0)
	# print("take:", list(take))


if __name__ == "__main__":
	main()
