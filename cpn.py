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

    # MILPMax
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

    # AtLeastReachable + Binary Search
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

    # AtLeastReachable
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


    # Algorithm 2 (Reachable) in Complexity Analysis of Continuous Petri Nets
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

    # Requires MOD to run!
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

    # Requires MOD to run!
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


    # Read cpn from .cpn file
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
