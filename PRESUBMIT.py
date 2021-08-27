#
# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Top-level presubmit script for libwebp2.

See https://dev.chromium.org/developers/how-tos/depottools/presubmit-scripts for
details on the presubmit API built into depot_tools.
"""

import subprocess2
USE_PYTHON3 = True
_INCLUDE_MAN_FILES_ONLY = [r"man/.+\.1$"]
_LIBWEBP2_MAX_LINE_LENGTH = 80


def _RunManCmd(input_api, output_api, man_file):
    cmd = ["man", "--warnings", "-EUTF-8", "-l", "-Tutf8", "-Z", man_file]
    name = "Check %s file." % man_file
    start = input_api.time.time()
    output, _ = subprocess2.communicate(cmd,
                                        stdout=None,
                                        stderr=subprocess2.PIPE,
                                        universal_newlines=True)
    duration = input_api.time.time() - start
    if output[1]:
        return output_api.PresubmitError(
            "%s\n%s (%4.2fs) failed\n%s" %
            (name, " ".join(cmd), duration, output[1]))
    return output_api.PresubmitResult("%s\n%s (%4.2fs)\n" %
                                      (name, " ".join(cmd), duration))


def _CheckManFiles(input_api, output_api):
    """Makes sure that libwebp2/ man files are clean"""

    man_sources = lambda x: input_api.FilterSourceFile(
        x, files_to_check=_INCLUDE_MAN_FILES_ONLY, files_to_skip=None)

    affected_man_files = input_api.change.AffectedFiles(
        file_filter=man_sources)
    results = [
        _RunManCmd(input_api, output_api, man_file.AbsoluteLocalPath())
        for man_file in affected_man_files
    ]
    return results


def _CommonChecks(input_api, output_api):
    results = []
    results.extend(
        input_api.canned_checks.CheckChangeHasNoCrAndHasOnlyOneEol(
            input_api, output_api))
    results.extend(
        input_api.canned_checks.CheckChangeHasNoTabs(input_api, output_api))
    results.extend(
        input_api.canned_checks.CheckChangeHasNoStrayWhitespace(
            input_api, output_api))
    results.extend(
        input_api.canned_checks.CheckLongLines(
            input_api, output_api, maxlen=_LIBWEBP2_MAX_LINE_LENGTH))
    results.extend(
        input_api.canned_checks.CheckPatchFormatted(input_api,
                                                    output_api,
                                                    check_clang_format=False,
                                                    check_python=True))
    return results


def CheckChangeOnUpload(input_api, output_api):
    results = []
    results.extend(_CommonChecks(input_api, output_api))
    results.extend(_CheckManFiles(input_api, output_api))
    return results


def CheckChangeOnCommit(input_api, output_api):
    results = []
    results.extend(_CommonChecks(input_api, output_api))
    results.extend(_CheckManFiles(input_api, output_api))
    return results
