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
_BASH_INDENTATION = "2"
_INCLUDE_BASH_FILES_ONLY = [r".*\.sh$"]
_INCLUDE_MAN_FILES_ONLY = [r"man/.+\.1$"]
_LIBWEBP2_MAX_LINE_LENGTH = 80


def _GetFilesToSkip(input_api):
  return list(input_api.DEFAULT_FILES_TO_SKIP) + [
      r"\.pylintrc$",
  ]


def _RunManCmd(input_api, output_api, man_file):
  cmd = ["man", "--warnings", "-EUTF-8", "-l", "-Tutf8", "-Z", man_file]
  name = "Check %s file." % man_file
  start = input_api.time.time()
  output, _ = subprocess2.communicate(
      cmd, stdout=None, stderr=subprocess2.PIPE, universal_newlines=True)
  duration = input_api.time.time() - start
  if output[1]:
    return output_api.PresubmitError("%s\n%s (%4.2fs) failed\n%s" %
                                     (name, " ".join(cmd), duration, output[1]))
  return output_api.PresubmitResult("%s\n%s (%4.2fs)\n" %
                                    (name, " ".join(cmd), duration))


def _RunShellCheckCmd(input_api, output_api, bash_file):
  """shellcheck command wrapper."""
  cmd = ["shellcheck", "-x", "-oall", "-sbash", bash_file]
  name = "Check %s file." % bash_file
  start = input_api.time.time()
  output, rc = subprocess2.communicate(
      cmd, stdout=None, stderr=subprocess2.PIPE, universal_newlines=True)
  duration = input_api.time.time() - start
  if rc == 0:
    return output_api.PresubmitResult("%s\n%s (%4.2fs)\n" %
                                      (name, " ".join(cmd), duration))
  return output_api.PresubmitError("%s\n%s (%4.2fs) failed\n%s" %
                                   (name, " ".join(cmd), duration, output[1]))


def _RunShfmtCheckCmd(input_api, output_api, bash_file):
  """shfmt command wrapper."""
  cmd = [
      "shfmt", "-i", _BASH_INDENTATION, "-bn", "-ci", "-sr", "-kp", "-d",
      bash_file
  ]
  name = "Check %s file." % bash_file
  start = input_api.time.time()
  output, rc = subprocess2.communicate(
      cmd, stdout=None, stderr=subprocess2.PIPE, universal_newlines=True)
  duration = input_api.time.time() - start
  if rc == 0:
    return output_api.PresubmitResult("%s\n%s (%4.2fs)\n" %
                                      (name, " ".join(cmd), duration))
  return output_api.PresubmitError("%s\n%s (%4.2fs) failed\n%s" %
                                   (name, " ".join(cmd), duration, output[1]))


def _RunCmdOnCheckedFiles(input_api, output_api, run_cmd, files_to_check):
  """Ensure that libwebp2/ files are clean."""
  file_filter = lambda x: input_api.FilterSourceFile(
      x, files_to_check=files_to_check, files_to_skip=None)

  affected_files = input_api.change.AffectedFiles(file_filter=file_filter)
  results = [
      run_cmd(input_api, output_api, f.AbsoluteLocalPath())
      for f in affected_files
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

  source_file_filter = lambda x: input_api.FilterSourceFile(
      x, files_to_skip=_GetFilesToSkip(input_api))
  results.extend(
      input_api.canned_checks.CheckLongLines(
          input_api,
          output_api,
          maxlen=_LIBWEBP2_MAX_LINE_LENGTH,
          source_file_filter=source_file_filter))

  results.extend(
      input_api.canned_checks.CheckPatchFormatted(
          input_api,
          output_api,
          check_clang_format=False,
          check_python=True,
          result_factory=output_api.PresubmitError))
  results.extend(
      _RunCmdOnCheckedFiles(input_api, output_api, _RunManCmd,
                            _INCLUDE_MAN_FILES_ONLY))
  # Run pylint.
  results.extend(
      input_api.canned_checks.RunPylint(
          input_api, output_api, pylintrc=".pylintrc", version="2.7"))

  # Binaries shellcheck and shfmt are not installed in depot_tools.
  # Installation is needed
  try:
    subprocess2.communicate(["shellcheck", "--version"])
    results.extend(
        _RunCmdOnCheckedFiles(input_api, output_api, _RunShellCheckCmd,
                              _INCLUDE_BASH_FILES_ONLY))
    print("shfmt")
    subprocess2.communicate(["shfmt", "-version"])
    results.extend(
        _RunCmdOnCheckedFiles(input_api, output_api, _RunShfmtCheckCmd,
                              _INCLUDE_BASH_FILES_ONLY))
  except OSError as os_error:
    results.append(
        output_api.PresubmitPromptWarning(
            "%s\nPlease install missing binaries locally." % os_error.args[0]))
  return results


def CheckChangeOnUpload(input_api, output_api):
  results = []
  results.extend(_CommonChecks(input_api, output_api))
  return results


def CheckChangeOnCommit(input_api, output_api):
  results = []
  results.extend(_CommonChecks(input_api, output_api))
  return results
