import sys
import networkx as nx

from time import time
from argparse import ArgumentParser
from cpn import cpn
from prettify import red, green, warn, blue
from random import sample, randint

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('cpn_file')
    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    parser.add_argument("--first_n_sols", '-f', default=1)
    parser.add_argument("--exact", '-e', action='store_true', default=False)
    parser.add_argument("--min", '-min', action='store_true', default=False)
    parser.add_argument("--max", '-max', action='store_true', default=False)
    parser.add_argument("--fragility", '-fr', action='store_true', default=False)
    parser.add_argument("--time_test", "-tt", nargs=2, default=0)
    return parser.parse_args(sys.argv[1:])

def build_arg(net):
    arg = net.build_arg()

    print(f"ARG {green('nodes')}: {arg.nodes}")
    print(f"ARG {green('edges')}")
    count=0
    for (u,v,attrs) in arg.edges(data=True):
        print(f"ID: {count}, Type: {arg[u][v]['type']}, label: {arg[u][v]['debug_label']}")
        count+=1
        """ if v.data==[] or v.data==['Formaldehyde']: # or (len(v.data)<=2 and 'Formaldehyde' in v.data):
            print(f"ID: {count}, Type: {arg[u][v]['type']}, label: {arg[u][v]['debug_label']}")
            count+=1 """

    return arg


def build_srt(net, arg):
    if list(arg.nodes) == []:
        print(red("Cannot build SRT as ARG is empty."))
        return

    srt = net.build_srt(arg)

    # print(f'SRT {red("nodes")}: {srt.nodes(data=False)}')
    print(f"SRT {green('edges')}")
    count=0
    for (u,v,attrs) in srt.edges(data=True):
        print(f"ID: {count}, label: {srt[u][v]['label']}")
        # print(f"\texpansion: {u}->{v.data[0].data}")
        count+=1
    return srt

# Algorithms (alg) classification:
#   - "exact" for exact reachable
#   - "min" for mininum reachable
#   - "max" for maximize goal compound
def calculate_reachable(start_marking, end_marking, net, lim_reachable=False, algs=["exact"], take_first_n=1):
    print(f"\nStarting from marking {start_marking}, is it possible to get to marking {end_marking}?")
    if set(start_marking) == set(end_marking):
        print_reachability_conc(True)

    times = {}

    if "exact" in algs:
        print("\nREACHABLE: EXACT GOAL MARKING?\n===========================")
        start = time()
        (is_reachable, sol) = net.try_exact_reachable(sm=start_marking, em=end_marking)
        end = time()
        times["exa"]= end-start
        print_reachability_conc(is_reachable)
    
    if "min" in algs:
        print("\nMIN-REACHABLE: GOAL MARKING?\n===========================")
        start = time()
        max_reachable_mass = net.max_min(sm=start_marking, em=end_marking, lim_reachability=True)
        end = time()
        times["min"] = end-start
        print(f"Maximum reachable mass is: {max_reachable_mass}")

    if "max" in algs:
        print("\nMAXIMIZE GOAL COMPOUND\n===========================")
        start = time()
        (is_reachable, sol, obj_val) = net.maximize_goal_compound(sm=start_marking, em=end_marking, first_n_sols=take_first_n)
        end = time()
        times["max"] = end-start

    return (times)

# times must be submitted in order! exact -> min -> max
def print_time_comparison(times):
    timecomp = ["-"*20]
    timecomp.append(f"| alg |   time{' '*5}|")
    timecomp.extend([f"| {alg} | {format_number(timeval, 10)} |" for (alg, timeval) in times.items()])
    timecomp.append("-"*20)
    print("\n".join(timecomp))

def format_number(number, num_spaces):
    return f"{number:{num_spaces}.5f}"

def print_reachability_conc(is_reachable):
    if is_reachable:
        print(green(f"Reachable!"))
    else:
        print(red(f"Not reachable.")) 

def time_test(net, algs, lim_reachable=True, percent=0.5, num_iters=10):
    places = net.places
    alg_sums = {alg[:3]:0 for alg in algs}
    for _ in range(num_iters):
        sm = [(p, 1.0) for p in sample(places, round(percent*len(places)))]
        em = [(places[randint(0,len(places)-1)], 1.0)]
        times = calculate_reachable(sm, em, net, lim_reachable=lim_reachable, algs=algs, take_first_n=1)
        alg_sums = {alg: alg_sums[alg]+times[alg] for alg in alg_sums.keys()}

    alg_sums = {alg: alg_sums[alg]/num_iters for alg in alg_sums.keys()}
    print_time_comparison(alg_sums)


def main():
    args = parse_args()
    net = cpn(file_name=args.cpn_file, verbose=args.verbose)

    algs = []
    if args.exact: algs.append("exact")
    if args.min: algs.append("min")
    if args.max: algs.append("max")
    if len(algs)==0: algs.append("exact")

    #sm = [("ribulose-5-phosphate", 1.0), ("h2o", 1.0)]
    #em = [("fructose-6-phosphate", 1.0)]

    if args.fragility:
        net.calculate_fragility(sm, em, alg='max', percent=0.25, n=5)
    elif args.time_test:
        time_test(net, algs=algs, lim_reachable=True, percent=float(args.time_test[1]), num_iters=int(args.time_test[0]))
    else:
        calculate_reachable(sm, em, net, lim_reachable=True, algs=algs, take_first_n=int(args.first_n_sols))

    #arg = build_arg(net)
    # srt = build_arg(net, arg)


    

    


if __name__=="__main__":
    main()