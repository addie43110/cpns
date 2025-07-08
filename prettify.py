class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def header(s: str) -> str:
    return f'{bcolors.HEADER}{s}{bcolors.ENDC}'

def blue(s: str) -> str:
    return f'{bcolors.OKBLUE}{s}{bcolors.ENDC}'

def green(s: str) -> str:
    return f'{bcolors.OKGREEN}{s}{bcolors.ENDC}'

def warn(s: str) -> str:
    return f'{bcolors.WARNING}{s}{bcolors.ENDC}'

def red(s: str) -> str:
    return f'{bcolors.FAIL}{s}{bcolors.ENDC}'

def bold(s: str) -> str:
    return f'{bcolors.BOLD}{s}{bcolors.ENDC}'

def underline(s: str) -> str:
    return f'{bcolors.UNDERLINE}{s}{bcolors.ENDC}'