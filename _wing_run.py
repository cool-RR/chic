# This file is a hack needed because of a bug in Wing's code analysis. If you don't use Wing, feel
# free to ignore it.
import sys

import chic.run

if __name__ == '__main__':
    try:
        chic.run.main()
    except SystemExit as system_exit:
        if system_exit.code != 0:
            sys.excepthook(*sys.exc_info())