"""A module with our customization for stdout to include a file log."""
import sys
import datetime
import re


class _StdoutLogger(object):
    """A class that simulates sys.stdout so it also prints to a file."""

    def __init__(self, file_name):
        """Initialize with the file name where we want to log."""
        super().__init__()
        # retrieve the original stdout file handle
        if isinstance(sys.stdout, _StdoutLogger):
            self._stdout_original = sys.stdout._stdout_original
            self._last_name = sys.stdout._last_name
            self._last_msg_len = sys.stdout._last_msg_len
        else:
            self._stdout_original = sys.stdout
            self._last_name = ""
            self._last_msg_len = 0

        # this will be the file where we also log
        self._stdout_file = file_name

    def write(self, msg, name=None):
        """Write to file and console.

        Args:
            msg (str): string to be printed
            name (str, optional): current prefix for message. Defaults to None.
        """
        # double check that sys.stdout is pointing to this instance
        if sys.stdout != self:
            sys.stdout = self

        # remove any type of EOL characters at the end of the message
        msg = re.sub("[\\n|\\r|\\b]*$", "", str(msg))

        # check for new name
        if name is None:
            name = self._last_name
        else:
            # modify name and check whether it's generic (name string empty)
            name = re.sub("\\$|{|}", "", name)

        is_generic = name == ""

        # is msg is empty and it's the same name as before, let's skip it.
        if msg == "" and name == self._last_name:
            return

        # generate name string and end of line character
        if is_generic:
            name_str = ""
            end_str = "\n"
        else:
            name_str = "{:19}: ".format(name)
            end_str = ""

        # generate full message
        msg = name_str + str(msg)

        # generate symbols for start_str. This depends on what type of
        # print we had before (specified by the last name)
        start_str = "\r"
        if name != self._last_name and self._last_name != "":
            start_str = "\n" + start_str

        # simulate carriage return with delete
        if start_str == "\r" and end_str != "\n":
            print(
                start_str + " " * self._last_msg_len,
                file=self._stdout_original,
                end=end_str,
            )

        # print with desired start sequence, name string, and end symbol
        print(start_str + msg, end=end_str, file=self._stdout_original)

        # also write to log file
        time_tag = datetime.datetime.utcnow().strftime("%Y-%m-%d, %H:%M:%S.%f")
        with open(self._stdout_file, "a") as logfile:
            print(f"{time_tag}: {msg}", file=logfile)

        # store last_name
        self._last_name = name

        # also store last message length
        self._last_msg_len = len(msg)

    def flush(self):
        """Flush console and file."""
        self._stdout_original.flush()


def setup_stdout(log_file):
    """Set up stdout logger with this function."""
    # get an instance of the stdout logger
    stdout_logger = _StdoutLogger(log_file)

    # set it to be the standard logger
    sys.stdout = stdout_logger

    return stdout_logger
