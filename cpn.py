import networkx as nx
import re
import ast
import mod
import random

from pulp import *

from prettify import red, warn, green, blue
from typing import List, Tuple, Dict
from more_itertools import powerset
from itertools import product
from scipy import optimize
from tqdm import tqdm
from os import listdir, path
from pathlib import Path

class cpn:
    def __init__(self, file_name: str = None, graph: nx.Graph=None, verbose: bool = False):
        self._file_name: str = file_name
        self._verbose: bool = verbose

        # extract number of repeats from filename
        if file_name:
            match = re.search(r"repeat-(\d)", file_name)
            if match:
                self._num_repeats: int = int(match.group(1))
            else:
                self._num_repeats = False

        if file_name:
            (success, g, ps, ts, initial_marking) = self.load_cpn()
            if success: 
                self._graph: nx.Graph = g
                self._places = ps
                self._transitions = ts
                self._init_marking = [(p, tm) for (p, tm) in initial_marking]
            else: print(red("CPN file could not be read."))
        else:
            self._graph: nx.Graph = graph
            self._places = [p for (p,_) in graph.nodes(data=True) if graph.nodes[p]['type']=='place']
            self._transitions = [t for (t,_) in graph.nodes(data=True) if graph.nodes[t]['type']=='transition']
            self._init_marking = [(p,attrs['tokens'])  for (p,attrs) in graph.nodes(data=True) \
                                                                if graph.nodes[p]['type']=='place' \
                                                                and graph.nodes[p]['tokens']>0]

        self._p_ind_map = {p : i for (p, i) in zip(self._places, list(range(len(self._places))))}
        self._t_ind_map = {t:i for (t,i) in zip(self._transitions, list(range(len(self._transitions))))}

        self._pre = self.build_pre_incidence(self._transitions)
        self._post = self.build_post_incidence(self._transitions)
        self._incidence_matrix = self.build_incidence_matrix(self._pre, self._post)

        self._clos_dict = {}


        # self.calculate_mode(self._init_marking)
        # mode_dict = self.sort_markings_by_mode(self._init_marking)
        # self.build_srt(mode_dict)
    
    @property
    def places(self):
        return self._places

    @property
    def transitions(self):
        return self._transitions

    # percent is number of transitions to remove / total number transitions
    # 0 <= percent <= 1
    def remove_random_n_transitions(self, percent):
        if percent > 1 or percent < 0:
            print(red(f"Cannot remove {percent}% of transitions."))
            return nx.Graph()

        removed_trans = random.sample(self._transitions, int(round(len(self._transitions)*percent)))
        new_graph = self._graph.copy()
        for t in removed_trans:
            new_graph.remove_node(t)

        new_net = cpn(graph=new_graph)
        return new_net

    # n is number of times to run the algorithm
    # percent is percentage of transitions to randomly remove
    def calculate_fragility(self, sm, em, alg='min', percent=0.5, n=5):
        num_succ = 0
        sum_obj_val = 0
        for i in range(n):
            test_net = self.remove_random_n_transitions(percent)
            if alg == 'max':
                (succ, sol, obj_val) = test_net.maximize_goal_compound(sm, em)
                if succ:
                    num_succ += 1
                    sum_obj_val += obj_val
            elif alg == 'min':
                obj_val = test_net.max_min(sm, em, lim_reachability=True)
                sum_obj_val += obj_val
                num_succ += 1
            
        print(f"\n\n=====================\nNumber of successes: {num_succ}")
        print(f"Total runs: {n}")
        print(f"Success rate: {float(num_succ)/n}")
        if num_succ > 0:
            print(f"Average objective value: {float(sum_obj_val)/num_succ}")


    def build_pre_incidence(self, transition_set):
        c = [[ 0 for i in range(len(self._transitions))] for j in range(len(self._places))]
        
        for p in self._places:
            for u,v,attrs in self._graph.out_edges(p, data=True):
                if v in transition_set:
                    c[self._p_ind_map[p]][self._t_ind_map[v]] = attrs['weight']

        #if self._verbose: print(f"\nPre-incidence: {c}\n")
        return c

    def build_post_incidence(self, transition_set):
        c = [[ 0 for i in range(len(self._transitions))] for j in range(len(self._places))]
        
        for p in self._places:
            for u,v,attrs in self._graph.in_edges(p, data=True):
                if u in transition_set:
                    c[self._p_ind_map[p]][self._t_ind_map[u]] = attrs['weight']

        # if self._verbose: print(f"\nPost-incidence: {c}\n")
        return c
    
    def build_incidence_matrix(self, pre, post):
        c = [[ post[j][i]-pre[j][i] \
                for i in range(len(pre[0]))] \
                for j in range(len(pre))]

        # if self._verbose: print(f"\nIncidence matrix: {c}\n")
        return c

    def build_arg(self):
        if self._init_marking == []:
            print(red(f"Cannot build ARG: initial marking empty."))
            return nx.DiGraph()

        if self._verbose:
            print(green("BUILDING ARG..."))

        (reachable_ps,_,_,_) = self.calculate_mode(self._init_marking)
        ps_rps = list(powerset(reachable_ps))
        arg_vertices = [vertex(list(x), label=f"arg_node_{i}") for (x,i) in zip(ps_rps, list(range(len(ps_rps))))]
        arg = nx.DiGraph()

        pbar = tqdm(total=len(arg_vertices)*len(arg_vertices))

        for p_prime in arg_vertices:
            pp_data = p_prime.data
            print(f"p_prime: {pp_data}")
            for p_double_prime in arg_vertices:
                if p_double_prime is p_prime:
                    pbar.update(1)
                    continue
                pdp_data = p_double_prime.data
                print(f"p__double_prime: {pdp_data}")
                if set(pp_data)!=set(pdp_data):
                    (_, p_prime_MFS) = self.fireable_p_prime(pp_data, self._transitions)
                    (_, p_double_prime_MFS) = self.fireable_p_prime(pdp_data, self._transitions)
                    if set(p_prime_MFS) == set(p_double_prime_MFS):
                        t_prime = p_prime_MFS
                        t_prime_out = []
                        for t in t_prime:
                            t_prime_out.extend(list(self._graph.successors(t)))
                        t_prime_out = list(set(t_prime_out))
                        t_prime_in = []
                        for t in t_prime:
                            t_prime_in.extend(list(self._graph.predecessors(t)))
                        t_prime_in = list(set(t_prime_in))
                        p_prime_union_t_prime_out = set(pp_data) | set(t_prime_out)
                        pp_minus_tp_in = set(pp_data) - set(t_prime_in)
                        if pp_minus_tp_in <= set(pdp_data) and set(pdp_data) <= p_prime_union_t_prime_out:
                            arg.add_edge(p_prime, p_double_prime, type='anonymous', debug_label=f"{p_prime}->{p_double_prime}")
                    elif set(p_double_prime_MFS)<set(p_prime_MFS) and set(pdp_data)<set(pp_data):
                        t_p_candidates = [list(x) for x in list(powerset(self._transitions))]
                        # TODO: compute the mode for each node
                        t_prime = False
                        for t_set in t_p_candidates:
                            in_places = []
                            out_places = []
                            for t in t_set:
                                in_places.extend(list(self._graph.predecessors(t)))
                                out_places.extend(list(self._graph.successors(t)))
                            in_out = set(in_places) | set(out_places)
                            pp_minus_in = set(pp_data) - set(in_places)

                            if set(in_out) <= set(pp_data) and pp_minus_in <= set(pdp_data):
                                t_prime = t_set
                                break
                        if t_prime:
                            # if t_prime makes a cycle, then only if that cycle consumes more than
                            # it produces can t_prime be a border edge.\
                            # TODO: needs work
                            """ potential_cycle_graph = nx.DiGraph()
                            for trans in t_prime:
                                potential_cycle_graph.add_edges_from(self._graph.in_edges(trans, data=True))
                                potential_cycle_graph.add_edges_from(self._graph.out_edges(trans, data=True))

                            add_border_edge = True

                            try:
                                cycle = nx.find_cycle(potential_cycle_graph, orientation='original')
                                print(green("cycle found"))
                                product = 1.0
                                for i in range(int(len(cycle)/2)):
                                    u1,v1,_ = cycle[2*i]
                                    u2,v2,_ = cycle[2*i+1]
                                    if u1 in self._places:
                                        denominator = self._graph.edges[u1,v1]['weight']
                                        nominator = self._graph.edges[u2,v2]['weight']
                                    else:
                                        denominator = self._graph.edges[u2,v2]['weight']
                                        nominator = self._graph.edges[u1,v1]['weight']
                                    product = product * (float(nominator)/denominator)
                                if product < 1:
                                    add_border_edge = True
                            except nx.NetworkXNoCycle:
                                add_border_edge = True
                                
                            if add_border_edge: """
                            arg.add_edge(p_prime, p_double_prime, type='border', label=str(t_prime), debug_label=f"{p_prime}->{t_prime}->{p_double_prime}")
                        
                pbar.update(1)
        pbar.close()
            
        return arg

    # m is a marking (single vertex in arg), returns a set of markings (vertex in srt)
    # all markings in r should share the same mode
    # calculating without token mass to start, as it is easier
    # RETURNS A LIST, NOT A VERTEX!!
    def closure(self, m, arg):

        if m in self._clos_dict:
            return self._clos_dict[m]

        clos = [m]
        stack = [m]
        visited = []

        # DFS through all adjacent anonymous edges
        while stack:
            arg_vertex = stack.pop()
            if arg_vertex in visited: continue
            visited.append(arg_vertex)

            marking_tokens = [(p, 1.0) for p in arg_vertex.data]
            (reachable_ps,_,_,_) = self.calculate_mode(marking_tokens)

            out_anon = [v for v in arg.successors(arg_vertex) if arg[arg_vertex][v]['type']=="anonymous" and set(arg_vertex.data) <= set(reachable_ps)]
            out_anon_stack = [v for v in out_anon if v not in stack and v not in visited]
            stack.extend(out_anon_stack)

            clos.extend(list(set(out_anon)))

        clos = list(set(clos))
        #for marking in clos:
        #    self._clos_dict[marking] = clos

        return list(set(clos))

    # r is a set of markings (set of vertices in arg -> single vertex in srt), returns a set of set of markings (multiple vertices to be made in srt)
    def succ(self, r, arg, parent_name):
        pbar = tqdm(total=len(r))
        srt_ve = []
        v_count = 0
        for m in r:
            border_edges = [(u,v) for (u,v) in arg.edges if u==m and v in arg.successors(m) and arg[m][v]['type']=='border']
            for (pp, pdp) in border_edges:
                # if pdp.data == []: continue
                transition = arg[pp][pdp]['label']
                new_vertex = vertex(self.closure(pdp, arg), label=f"{parent_name}.{v_count}")
                v_count+=1
                srt_ve.append((new_vertex, f"{parent_name}->{transition}->{new_vertex}"))
            pbar.update(1)
        pbar.close()
        return srt_ve

    def build_srt(self, arg) -> nx.Graph:
        if self._verbose:
            print(green("BUILDING SRT..."))

        if self._init_marking==[]:
            print(red("Cannot build SRT: initial marking empty."))
            return nx.DiGraph()

        m0 = vertex([p for (p,_) in self._init_marking], label="start_marking")
        srt = nx.DiGraph()
        
        root = vertex(self.closure(m0, arg), label="root")
        srt.add_node(root)

        queue = [root]
        visited = []
        while queue:
            curr_srt_vertex = queue.pop(0)
            if curr_srt_vertex in visited: continue
            visited.append(curr_srt_vertex)

            new_vertices = self.succ(curr_srt_vertex.data, arg, parent_name=curr_srt_vertex.label)

            for (v, t) in new_vertices:
                srt.add_edge(curr_srt_vertex, v, label=t)
                queue.append(v)
        
        return srt

    def directly_fireable(self, places):
        df_transitions = []
        for t in self._transitions:
            in_places = list(self._graph.predecessors(t))
            if set(in_places)<=set(places):
                df_transitions.append(t)
        return df_transitions

    # Algorithm 1 in Complexity Analysis of Continuous Petri Nets
    # Fraca and Haddad, pg. 11
    # marking of type [("p1", 1.0), ("p2", 0.5)]
    # t_prime: subset of transitions
    def fireable(self, marking, t_prime):
        t_double_prime = [] # subset of transitions
        p_prime = [p for (p,_) in marking] # subset of places
        witness = []
        while set(t_double_prime)!=set(t_prime):
            new = False
            for t in list(set(t_prime)-set(t_double_prime)):
                in_places = list(self._graph.predecessors(t))
                if set(in_places)<=set(p_prime):
                    out_places = list(self._graph.successors(t))
                    witness.append(f"{in_places}->{t}->{out_places}")
                    t_double_prime.append(t)
                    p_prime.extend(out_places)
                    new = True
            if not new:
                return (False, t_double_prime, None)
        return (True, t_double_prime, witness)

    # Algorithm 1 in Complexity Analysis of Continuous Petri Nets
    # Fraca and Haddad, pg. 11
    # p_prime of type ["p1", "p2"]
    # t_prime: subset of transitions
    def fireable_p_prime(self, p, t_prime):
        t_double_prime = [] # subset of transitions
        p_prime = [x for x in p]
        while set(t_double_prime)!=set(t_prime):
            new = False
            for t in list(set(t_prime)-set(t_double_prime)):
                in_places = list(self._graph.predecessors(t))
                if set(in_places)<=set(p_prime):
                    t_double_prime.append(t)
                    p_prime.extend(list(self._graph.successors(t)))
                    new = True
            if not new:
                return (False, t_double_prime)
        return (True, t_double_prime)

    def maximize_goal_compound(self, sm, em, forbidden_FS=[], first_n_sols = 1):
        verb = self._verbose
        # convert marking to a |P| length vector
        m0 = [0 for i in range(len(self._places))]
        m = [0 for i in range(len(self._places))]
        for (pl, token_mass) in sm:
            m0[self._p_ind_map[pl]] = token_mass

        milp = LpProblem("Maximize_Goal_Compound", sense=LpMaximize)
        t_vars = LpVariable.dicts("transition", self._transitions, lowBound=0, upBound=sum(m0), cat="Continuous")
        b_vars = LpVariable.dicts("selected", self._transitions, cat="Binary")
        HUGE_CONST = 1e9
        #HUGE_CONST = 1
        goal_place = em[0][0]
        # obj = [0 for _ in range(len(self._transitions))]
        obj = []
        for t in self._graph.predecessors(goal_place):
            obj.append((t_vars[t], self._graph.edges[t, goal_place]['weight']*HUGE_CONST))
            # obj[self._t_ind_map[t]] += self._graph.edges[t, goal_place]['weight']
        for t in self._graph.successors(goal_place):
            obj.append((t_vars[t], 0-self._graph.edges[goal_place, t]['weight']*HUGE_CONST))
            # obj[self._t_ind_map[t]] -= self._graph.edges[goal_place, t]['weight']

        obj.extend([(bv, -1) for bv in list(b_vars.values())])
        milp += lpSum([var*coef for (var, coef) in obj])

        ind_to_t = {i:t for (t,i) in self._t_ind_map.items()}

        # incidence (the maxmimum amount of token mass on any single place can be sum(m0))
        pre = self.build_pre_incidence(self._transitions)
        post = self.build_post_incidence(self._transitions)
        c_t_prime = self.build_incidence_matrix(pre, post)
        count = 0
        for line in c_t_prime:
            milp += lpSum([t_vars[ind_to_t[i]]*line[i] for i in range(len(line))]) >= (0-m0[count])
            count+=1
            # milp += lpSum([t_vars[ind_to_t[i]]*line[i] for i in range(len(line))]) <= sum(m0)

        # b_i - t_i < 1
        for t_i, b_i in zip(t_vars.values(), b_vars.values()):
            milp += b_i - t_i <= 0.99

        # t_i - b_i * c <= 0
        c = 1e10
        for t_i, b_i in zip(t_vars.values(), b_vars.values()):
            milp += t_i - b_i * c <= 0

        # 2(\SUM t \in T') < (\SUM t \in T) + |T'|
        constraints = []
        sums = []
        for fs in forbidden_FS:
            coefs = [1 if t in fs else -1 for t in self._transitions]
            milp += lpSum([b_vars[t]*coef for (t, coef) in zip(self._transitions, coefs)]) <= (len(fs)-1)

        # print(f"\n\n+++++++++++++++++++\nMILP\n{milp}")

        solver = getSolver('HiGHS', msg=0)

        max_tries = 400
        curr_iter = 0
        sol_count = 1
        while max_tries:
            milp.solve(solver)
            non_zero_transitions = [(t,round(t_i.value(),3)) for t, t_i in t_vars.items() if t_i.value()!=0]
            non_zero_booleans = [(b, b_v.value())for b,b_v in b_vars.items() if b_v.value()!=0]
            if verb:
                print(f"\nSOLUTION {red(curr_iter+1)}\n===================")
                print("transitions:", non_zero_transitions)
                print("booleans:", non_zero_booleans)
                print(f"obj value: {milp.objective.value()/HUGE_CONST}")
                print(f"final token mass on goal: {milp.objective.value()/HUGE_CONST+em[0][1]}")

            firing_set = [t for t,v in non_zero_transitions]
            (succ, _, witness) = self.fireable(sm, firing_set)
            if succ:
                print("\nPRINTING WITNESS....")
                print("\n".join(witness))
                print("")
                mod.post.summarySection(f"Solution {sol_count} -- obj val {round(milp.objective.value()/HUGE_CONST,5)}")
                #dg = self.turn_sol_into_DG(non_zero_transitions)
                #self.get_flow_solutions(dg)
                if sol_count>=first_n_sols:
                    return (True, non_zero_transitions, milp.objective.value()/HUGE_CONST)
                sol_count+=1
            # else:
            coefs = [1 if t in firing_set else -1 for t in self._transitions]
            milp += lpSum([b_vars[t]*coef for (t, coef) in zip(self._transitions, coefs)]) <= (len(firing_set)-1)
            
            max_tries-=1
            curr_iter+=1
        print(red(f"MAX TRIES ({max_tries}) EXHAUSTED -- NO SOLUTION"))
        return (False, False, max_tries)

    # ASSUMES only one support in em (end marking)
    def max_min(self, sm, em, lim_reachability):
        verb = self._verbose
        if len(em)!=1:
            print(red(f"Cannot find maximum token mass for goal marking with support size != 1: {em}"))
            return False

        ind_to_t = {i:t for (t,i) in self._t_ind_map.items()}
        goal_label, goal_mass = em[0]
        lower_bound = 0
        upper_bound = sum([mass for (_, mass) in sm])
        last_reachable_mass = None
        prev_mass = None
        test_mass = goal_mass/2
        while lower_bound<upper_bound:
            test_mass = round((upper_bound - lower_bound)/2 + lower_bound, 3)
            if test_mass == prev_mass: break
            test_em = [(goal_label, test_mass)]
            if verb: print(warn(f"Testing {test_mass} mass on goal compound '{goal_label}'..."))

            reachable = self.at_least_reachable(sm, test_em, lim_reachability=True)

            if reachable[0] is True:
                if verb: print(green(f"Can reach mass {test_mass}"))
                sol = [(ind_to_t[i], reachable[1][i]) for i in range(len(reachable[1]))]
                last_reachable_mass = test_mass
                lower_bound = test_mass
            else:
                if verb: print(red(f"Cannot reach mass {test_mass}"))
                upper_bound = test_mass

            prev_mass = test_mass

        if last_reachable_mass:
            print(f"Sol: {sol}")
            return last_reachable_mass

        return 0


    def at_least_reachable(self, sm, em, lim_reachability=True):
        # convert marking to a |P| length vector
        m0 = [0 for i in range(len(self._places))]
        m = [0 for i in range(len(self._places))]
        for (pl, token_mass) in sm:
            m0[self._p_ind_map[pl]] = token_mass
        for (pl, token_mass) in em:
            m[self._p_ind_map[pl]] = token_mass
        
        m_m0 = [m[i]-m0[i] for i in range(len(m))]
        t_prime = [t for t in self._transitions]
        solver = getSolver('HiGHS', msg=0)
        (_, maxFS, _) = self.fireable(marking=sm, t_prime=t_prime)
        ind_to_t = {i:t for (t,i) in self._t_ind_map.items()}

        while t_prime:
            nbsol = 0
            sol = [0 for i in range(len(self._transitions))]

            pre = self.build_pre_incidence(t_prime)
            post = self.build_post_incidence(t_prime)
            c_t_prime = self.build_incidence_matrix(pre, post)
            
            for t in t_prime:
                lp = LpProblem("Minimum_Goal_Token_Mass", sense=LpMinimize)
                t_vars = LpVariable.dicts("transition", self._transitions, lowBound=0, upBound=sum(m0), cat="Continuous")
                # objective
                lp += lpSum([t*1 for t in t_vars.values()])
                # constraints (incidence)
                count = 0
                for line in c_t_prime:
                    lp += lpSum([t_vars[ind_to_t[i]]*line[i] for i in range(len(line))]) >= (m_m0[count])
                    count+=1
                # constraint (v[t] > 0) (THIS IS AVOIDING A STRICT INEQUALITY; BEWARE!)
                lp += t_vars[t] >= 0.01
                # print(f"Running linear program...")
                status = lp.solve(solver)
                if status==0 or status==1:
                #     print(f"Status 0: success")
                #     print(f"Sample answer: {result.x}")
                    nbsol += 1
                    sol = [sol[i]+t_vars[ind_to_t[i]].value() for i in range(len(sol))]
            if nbsol == 0:
                return (False, False)

            sol = [sol[i]/nbsol for i in range(len(sol))]
            sol = [round(s, 5) for s in sol]

            #print(f"rounded sol: {sol}")
            t_prime = [ind_to_t[i] for i in range(len(sol)) if round(sol[i],5)>0]
            #print(f"t_prime: {t_prime}")
            copy_of_sol_as_t_prime = [ind_to_t[i] for i in range(len(sol)) if round(sol[i],5)>0]
            # only the places marked in m0 that are in-/out-places to T'
            m0_t_prime = []
            start_marking_places = [p for (p,_) in sm]
            for t in t_prime:
                for in_pl in self._graph.predecessors(t):
                    if in_pl in start_marking_places:
                        m0_t_prime.append((in_pl, m0[self._p_ind_map[in_pl]]))
                for out_pl in self._graph.successors(t):
                    if out_pl in start_marking_places:
                        m0_t_prime.append((out_pl, m0[self._p_ind_map[out_pl]]))

            #print(f"m0_t_prime: {set(m0_t_prime)}")
            (_, maxFS, _) = self.fireable(marking=set(m0_t_prime), t_prime=t_prime)
            #print(f"maxFS: {sorted(maxFS)}")
            # set intersection
            #print(f"outliers: {set(maxFS) ^ set(t_prime)}")
            t_prime = list(set(t_prime) & set(maxFS))
            #print(f"t_prime: {sorted(t_prime)}")

            if not lim_reachability:
                # delete for lim-reachability
                m_t_prime = []
                reverse_graph = self._graph.reverse()
                reverse_cpn = cpn(graph=reverse_graph)
                end_marking_places = [p for (p,_) in em]
                for t in t_prime:
                    for in_pl in reverse_graph.predecessors(t):
                        if in_pl in end_marking_places:
                            m_t_prime.append((in_pl, m[self._p_ind_map[in_pl]]))
                    for out_pl in reverse_graph.successors(t):
                        if out_pl in end_marking_places:
                            m_t_prime.append((out_pl, m[self._p_ind_map[out_pl]]))
                
                (_, maxFS,_) = reverse_cpn.fireable(marking=m_t_prime, t_prime=t_prime)
                t_prime = list(set(t_prime) & set(maxFS))
                # end of delete for lim-reachability

            if set(t_prime) == set(copy_of_sol_as_t_prime):
                #print(f"Sol: {sol}")
                #print(f"t_prime: {t_prime}")
                # self.turn_sol_into_DG(list(zip(self._transitions, sol)))
                return (True, sol)
        return (False, False)


    # Algorithm 2 in Complexity Analysis of Continuous Petri Nets
    # Fraca and Haddad, pg. 14
    def try_exact_reachable(self, sm, em, lim_reachability=True):

        # convert marking to a |P| length vector
        m0 = [0 for i in range(len(self._places))]
        m = [0 for i in range(len(self._places))]
        for (pl, token_mass) in sm:
            m0[self._p_ind_map[pl]] = token_mass
        for (pl, token_mass) in em:
            m[self._p_ind_map[pl]] = token_mass
        
        m_m0 = [m[i]-m0[i] for i in range(len(m))]
        t_prime = [t for t in self._transitions]
        obj = [1]*len(self._transitions)

        while t_prime:
            nbsol = 0
            sol = [0 for i in range(len(self._transitions))]
            (_, maxFS,_) = self.fireable(marking=sm, t_prime=t_prime)
            """ pre = self.build_pre_incidence(maxFS)
            post = self.build_post_incidence(maxFS)
            c_t_prime = self.build_incidence_matrix(pre, post) """
            pre = self.build_pre_incidence(t_prime)
            post = self.build_post_incidence(t_prime)
            c_t_prime = self.build_incidence_matrix(pre, post)
            for t in t_prime:
                # print(f"Running linear program...")
                # print(f"Objective function: {[1]*len(m)} @ v")
                # print(f"Contraints: {c_t_prime} @ v = {m_m0}")
                result = optimize.linprog(obj, A_eq=c_t_prime, b_eq=m_m0)
                if result.status == 0:
                #     print(f"Status 0: success")
                #     print(f"Sample answer: {result.x}")
                    nbsol += 1
                    sol = [sol[i]+result.x[i] for i in range(len(sol))]
            if nbsol == 0:
                return (False, False)
            
            # sol = [float(sol[i])/nbsol for i in range(len(sol))]
            sol = [float(sol[i])/len(t_prime) for i in range(len(sol))]
            print(f"normalized sol: {sol}")
            ind_to_t = {i:t for (t,i) in self._t_ind_map.items()}
            t_prime = [ind_to_t[i] for i in range(len(sol)) if sol[i]>0]
            print(f"t_prime: {t_prime}")
            copy_of_sol_as_t_prime = [ind_to_t[i] for i in range(len(sol)) if sol[i]>0]
            # only the places marked in m0 that are in-/out-places to T'
            m0_t_prime = []
            start_marking_places = [p for (p,_) in sm]
            for t in t_prime:
                for in_pl in self._graph.predecessors(t):
                    if in_pl in start_marking_places:
                        m0_t_prime.append((in_pl, m0[self._p_ind_map[in_pl]]))
                for out_pl in self._graph.successors(t):
                    if out_pl in start_marking_places:
                        m0_t_prime.append((out_pl, m0[self._p_ind_map[out_pl]]))

            print(f"m0_t_prime: {set(m0_t_prime)}")
            (_, maxFS,_) = self.fireable(marking=set(m0_t_prime), t_prime=t_prime)
            print(f"maxFS: {maxFS}")
            # set intersection
            print(f"outliers: {set(maxFS) ^ set(t_prime)}")
            t_prime = list(set(t_prime) & set(maxFS))
            print(t_prime)

            if not lim_reachability:
                # delete for lim-reachability
                m_t_prime = []
                reverse_graph = self._graph.reverse()
                reverse_cpn = cpn(graph=reverse_graph)
                end_marking_places = [p for (p,_) in em]
                for t in t_prime:
                    for in_pl in reverse_graph.predecessors(t):
                        if in_pl in end_marking_places:
                            m_t_prime.append((in_pl, m[self._p_ind_map[in_pl]]))
                    for out_pl in reverse_graph.successors(t):
                        if out_pl in end_marking_places:
                            m_t_prime.append((out_pl, m[self._p_ind_map[out_pl]]))
                
                (_, maxFS,_) = reverse_cpn.fireable(marking=m_t_prime, t_prime=t_prime)
                t_prime = list(set(t_prime) & set(maxFS))
                # end of delete for lim-reachability


            if set(t_prime) == set(copy_of_sol_as_t_prime):
                return (True, sol)
        return (False, False)


    def find_firing_sequences(self, mode_to_markings, larger_mode, smaller_mode):
        goal_markings = mode_to_markings[smaller_mode]
        start_markings = mode_to_markings[larger_mode]

        # no transitions need to be taken if we are already in our goal mode
        if larger_modes==smaller_mode:
            return []

        for sm in start_markings:
            for gm in goal_markings:
                if reachable(sm, gm):
                    return True


    def sort_markings_by_mode(self, marked: List[Tuple[str, float]]) -> Dict[str, List[List[str]]]:
        (reachable_ps,_,_,_) = self.calculate_mode(marked)
        possible_markings = list(powerset(reachable_ps))
        possible_markings = [list(subset) for subset in possible_markings]

        unique_markings = {}
        for pm in possible_markings:
            # note: calculate_mode does not consider exact token mass, only token_mass>0
            # so a token mass of 1.0 is added to the possible_markings as a placeholder
            (_,_,fireable_transitions,_) = self.calculate_mode(list(zip(pm, [1.0]*len(pm))))
            try:
                unique_markings[",".join(fireable_transitions)].append(pm)
            except KeyError:
                unique_markings[",".join(fireable_transitions)] = [pm]

        return unique_markings


    def calculate_mode(self, marking: List[Tuple[str, float]]):
        
        (fireable_transitions, reachable_ps) = self.calculate_fireable_iter(marking)
        unreachable_ps = [p for p in self._graph.nodes if self._graph.nodes[p]['type']=="place" and p not in reachable_ps]
        unfireable_transitions = [t for t in self._graph.nodes if self._graph.nodes[t]['type']=="transition" and t not in fireable_transitions]

        return (reachable_ps, unreachable_ps, fireable_transitions, unfireable_transitions)

    def calculate_fireable_iter(self, marking):
        fireable_transitions = []
        reachable_ps = [p for (p, _) in marking]
        stack = [p for (p, _) in marking]
        do_not_revisit = []
            
        while stack:
            place = stack.pop()

            for trans in self._graph.successors(place):
                if trans in do_not_revisit: continue
                if len(list(self._graph.predecessors(trans)))==1:
                    fireable_transitions.append(trans)
                    do_not_revisit.append(trans)
                    for pl in self._graph.successors(trans):
                        if pl not in reachable_ps:
                            reachable_ps.append(pl)
                        if pl not in stack:
                            stack.append(pl)
                else:
                    if all(p in reachable_ps for p in list(self._graph.predecessors(trans))):
                        fireable_transitions.append(trans)
                        do_not_revisit.append(trans)
                        for pl in self._graph.successors(trans):
                            if pl not in reachable_ps:
                                reachable_ps.append(pl)
                            if pl not in stack:
                                stack.append(pl)

        return (sorted(fireable_transitions), sorted(reachable_ps))

    def get_flow_solutions(self, dg):

        print(warn("\n\nMOD FLOW SOLUTION CALCULATING..."))

        for vertex in dg.vertices:
            if vertex.graph.name == "h2o": water = vertex
            if "ribulose" in vertex.graph.name: ribuloseP = vertex
            if "fructose" in vertex.graph.name: fructoseP = vertex

        flow = mod.hyperflow.Model(dg)
        flow.addSource(water)
        flow.addSource(ribuloseP)
        flow.addConstraint(mod.inFlow[ribuloseP] == 12)
        flow.addConstraint(mod.inFlow[water] == 2)
        flow.addSink(fructoseP)
        flow.addConstraint(mod.outFlow[fructoseP] == 10)
        flow.objectiveFunction = -mod.outFlow[fructoseP]
        for v in dg.vertices:
            flow.addSink(v.graph)
        flow.findSolutions(
            #verbosity=2
        )

        for s in flow.solutions:
            query = mod.causality.RealisabilityQuery(flow.dg)
            printer = mod.DGPrinter()
            printer.graphvizPrefix = 'layout = "dot";'
            printer.pushVertexVisible(lambda v: s.eval(mod.vertexFlow[v]) != 0)
            dag = query.findDAG(s)
            if dag:
                data = dag.getPrintData(False)
                dg.print(printer=printer, data=data)
            else:
                print(s, red("has no DAG, find catalysts"))
                for v in dg.vertices:
                    a = v.graph
                    if s.eval(mod.vertexFlow[a]) == 0:
                        continue
                    query = mod.causality.RealisabilityQuery(flow.dg)
                    query[a] = 1
                    dag = query.findDAG(s)
                    if dag:
                        mod.post.summarySection("Catalyst %s" % a.name)
                        data = dag.getPrintData(True)
                        dg.print(printer=printer, data=data)

        
    def turn_sol_into_DG(self, non_zero_transitions):

        # first, create the full solution DG
        if self._num_repeats:
            repeat = self._num_repeats
        else: 
            repeat = int(input("Please enter the number of repeats: "))
        graph_folder = path.join("./ppp/", "graphs")
        rule_folder = path.join("./ppp/", "rules")

        graphs = []
        for file_name in listdir(graph_folder):
            graph = mod.Graph.fromGMLFile(path.join(graph_folder, file_name))
            graph.name = file_name[:-4]
            if graph.name == "fructose-6-phosphate":
                fructose = graph
            graphs.append(graph)
        rules = [mod.Rule.fromGMLFile(path.join(rule_folder, file_name))
                    for file_name in listdir(rule_folder)]

        non_fructose = [g for g in graphs if g.name!="fructose-6-phosphate"]
        strat = (
            mod.addSubset(non_fructose)
            >> mod.repeat[repeat] (
                rules
            )
        )

        dg = mod.DG(graphDatabase=graphs)
        dg.build().execute(strat)

        t_count = 0
        sol_hyperedges = []
        print_products = []
        for hyperedge in dg.edges:
            t_name = f"t{t_count}"
            if any([t_name==name for name,_ in non_zero_transitions]):
                sol_hyperedges.append(hyperedge)

                for p in hyperedge.sources:
                    if p not in print_products:
                        print_products.append(p)
                
                for p in hyperedge.targets:
                    if p not in print_products:
                        print_products.append(p)
            t_count+=1

        # dg.dump(f"./ppp-repeat-4-dump.dg")

        # create a new DG with ONLY the solution products and hyperedges
        solDG = mod.DG(graphDatabase=[p.graph for p in print_products])
        with solDG.build() as b:
            for hyperedge in sol_hyperedges:
                b.addHyperEdge(hyperedge)

        p = mod.DGPrinter()
        p.pushEdgeVisible(lambda e: e in sol_hyperedges)
        p.pushVertexVisible(lambda prod: prod in print_products)
        p.graphvizPrefix = 'layout = "dot";'
        print("Printing solution derivation graph....\n")
        dg.print(p)

        p = mod.DGPrinter()
        p.graphvizPrefix = 'layout = "dot";'
        solDG.print(p)

        return solDG


    def load_cpn(self): 
        g = nx.DiGraph()
        places = []
        transitions = []
        marking = []

        with open(self._file_name) as f:
            while True:
                line = f.readline()
                if line == '': break

                if line.strip()=='': continue

                header = line.split()[0]

                if header=="name":
                    match = re.search(r"\"(.+)\"", line)
                    if match:
                        g.graph['name'] = match.group(1)
                        if self._verbose: print(f"reading name: {match.group(1)}")
                    else:
                        print(red("Name malformed."))
                        return False

                if header=="nodes":
                    line = f.readline().strip()
                    while line!="]":
                        if line == '':
                            line = f.readline().strip()
                            continue

                        if line[:5]=="place":
                            m = re.search(r"\"(.+)\"", line)
                            if m:
                                g.add_node(m.group(1), type="place", tokens=0)
                                places.append(m.group(1))
                            else: 
                                print(red("Reading nodes: place malformed."))
                                return False
                        elif line[:5]=="trans":
                            m = re.search(r"\"(.+)\"", line)
                            if m:
                                g.add_node(m.group(1), type="transition")
                                transitions.append(m.group(1))
                            else:
                                print(red("Reading nodes: transition malformed."))
                                return False
                        line = f.readline().strip()
                    if self._verbose: print(f"nodes: {g.nodes(data=False)}")

                if header=="incidence":
                    edge = f.readline().strip()
                    while edge!="]":
                        if edge == '':
                            edge = f.readline().strip()
                            continue
                        tokens = edge.split()

                        in_edges = []
                        multiplier = 1
                        for token in tokens:
                            if token in g.nodes:
                                if g.nodes[token]['type']=='place':
                                    in_edges.append((token, multiplier))
                                    multiplier = 1
                                else:
                                    transition = token
                                    break
                            else:
                                multiplier = float(token)

                        out_edges = []
                        reached_out_edges = False
                        multiplier = 1
                        for token in tokens:
                            if token in g.nodes:
                                if reached_out_edges and g.nodes[token]['type']=='place':
                                    out_edges.append((token, multiplier))
                                    multiplier = 1
                                elif g.nodes[token]['type']=='transition':
                                    reached_out_edges = True
                            elif reached_out_edges:
                                multiplier = float(token)

                        for ie, mult in in_edges:
                            g.add_edge(ie, transition, weight=mult)

                        for oe, mult in out_edges:
                            g.add_edge(transition, oe, weight=mult)
                        
                        edge = f.readline().strip()
                        if self._verbose: print(f"transition read: {in_edges}->{transition}->{out_edges}")

                if header=="marking":
                    tokens = f.readline().strip()
                    while tokens!="]":
                        if tokens == '':
                            tokens = f.readline().strip()
                            continue

                        tokens = tokens.split()
                        place = tokens[0]
                        token_mass = 1.0

                        if len(tokens)>1:
                            token_mass = float(tokens[1])
                            
                        if place not in g.nodes:
                            print(red("Place marked before defined."))
                            return False

                        if token_mass > 0:
                            marking.append((place, token_mass))
                            if self._verbose: print(f"place marked: {place} with {token_mass} token mass.")
                        tokens = f.readline().strip()

            # end while
            return (True, g, places, transitions, marking)

    
def get_hamming_one_pairs(old_markings, new_markings):
    pairs = []
    for o_m in old_markings:
        for n_m in new_markings:
            # symmetric set difference
            if (len(list(set(o_m) ^ set(n_m)))) <= 2:
                pairs.append((o_m, n_m))

    print(f"\nPairs: {pairs}")
    return pairs

class vertex:
    def __init__(self, data, label="unlabelled", isList=True):
        self._data = sorted(list(set(data))) if isList else data
        self._name = (str(self._data))
        self._label = str(label) if len(self._name)>20 else self._name

    def __str__(self):
        return self._label

    def __repr__(self):
        return self._label

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._name == other.name
        else:
            return False

    def __hash__(self):
        return hash(self._name)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self._name < other.name

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    @property
    def num_markings(self):
        return len(self._data)

    @property
    def preview(self):
        if len(self._name)>20:
            return self._name[:20]
        else:
            return self._name

    @property
    def label(self):
        return self._label