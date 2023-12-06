#!/usr/bin/env python3
#
#  projects.py
"""
Pipeline for creating projects from raw datafiles.
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

__all__ = ("projects", )


@click.argument("projects_toml", default="projects.toml")
@click.command()
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


if __name__ == "__main__":
	projects()
