from nltk import CFG, Production, Nonterminal
from nltk.tree import Tree
from typing import List, Tuple, Callable, Dict, Any, Union, Optional, Iterable
from itertools import product


class CykParser:
    def __init__(self, grammar: CFG, longest_unary_chain=6):
        """

        :param grammar: a context free grammar of the type nltk.CFG
        """
        self.grammar = grammar
        self.grammar_cnf = None
        self.removed_unary_rules = set()
        self.longest_chain_length = longest_unary_chain

        # if nonterminal_counter > 0, then the added nonterminals are 0, 1, ..., nonterminal_counter-1
        self.nonterminal_counter = 0
        self.added_nonterminals = []
        self.generate_cnf()
        self.unary_chains = self.generate_unary_chains()

    def parse(self, sentence: str) -> List[Tree]:
        """

        :param sentence: the string to parse
        :return: all possible trees
        """
        table = self.fill_cyk_table(sentence)
        # keep only the start symbol in the top-right cell
        trees = self.build_trees(table, 0, len(sentence.split(' '))-1)
        restored_trees = self.cnf_to_cfg(trees)
        # convert labels and leaves of the tree to strings so they can be pretty-printed
        trees = [Tree.fromstring(str(t)) for t in restored_trees if t.label() == self.grammar.start()]
        return trees

    def fill_cyk_table(self, sentence: str) -> List[Any]:
        """
        The CYK algorithm body
        :param sentence: tokenized sentence
        :return:
        """
        if sentence[-1] == '.':
            sentence = sentence[:-1]
        sentence = sentence.split(' ')
        M = len(sentence)
        rules = self.grammar_cnf.productions()
        # initialize M*M*0 table
        table = [[{} for i in range(M)] for i in range(M)]

        # fill in the diagonal from top-left to bottom-right
        for m in range(M):
            for rule in rules:
                if rule.rhs()[0] == sentence[m]:
                    table[m][m][rule.lhs()] = [sentence[m]]

        # fill in the rest of the table one diagonal at a time
        # l: constituent length
        for l in range(1, M):
            # m: left endpoint (equivalently, right endpoint)
            for m in range(M-l):
                # k: split point
                for k in range(m, m+l):
                    left = table[m][k]
                    right = table[k+1][m+l]
                    for rule in rules:
                        if len(rule.rhs()) == 2 and rule.rhs()[0] in left and rule.rhs()[1] in right:
                            table[m][m+l][rule.lhs()] = k
        return table

    def build_trees(self, table: List[Any], i, j) -> List[Tree]:
        if i == j:
            # return all rules that generate the word
            return [Tree(lhs, table[i][j][lhs]) for lhs in table[i][j]]

        result = []
        # iterate over all rules
        for lhs in table[i][j]:
            k = table[i][j][lhs]
            left_subtrees = self.build_trees(table, i, k)
            right_subtrees = self.build_trees(table, k+1, j)
            for left in left_subtrees:
                left_root = left.label()
                for right in right_subtrees:
                    right_root = right.label()
                    production = Production(lhs, [left_root, right_root])
                    if production in self.grammar_cnf.productions():
                        result.append(Tree(lhs, [left, right]))
        return result

    def cnf_to_cfg(self, trees: List[Tree]):
        """
        Convert the input trees of CNF into the original context free grammar
        :param trees: syntax trees
        :return:
        """
        step1 = [self.remove_added_nonterminals(tree) for tree in trees]
        step2 = self.restore_unary_rules(step1)
        return step2

    def generate_unary_chains(self) -> List[Tree]:
        """
        generate all possible chains for unary rules with length > 2
        e.g.
        if we have A -> B, B -> C, C -> D, D -> E F
        the result will be
        [
        [A, B, C, D, [E, F]],
        [A, B, C, [D]],
        [A, B, [C]],
        [B, C, D, [E, F]],
        [B, C, [D]],
        [C, D, [E, F]],
        ]
        :return: list of all possible chains of unary rules
        """
        extend_chains = True
        chains = [[rule.lhs(), rule.rhs()] for rule in self.removed_unary_rules]
        original_chains = list(chains)
        while extend_chains:
            new_chains = list(chains)
            for chain in chains:
                if len(chain[-1]) == 1:
                    for other_chain in chains:
                        if chain[-1][0] == other_chain[0]:
                            extended_chain = chain[:-1] + other_chain
                            if extended_chain not in new_chains and len(extended_chain) <= self.longest_chain_length:
                                new_chains.append(extended_chain)
            if len(new_chains) == len(chains):
                extend_chains = False
            chains = new_chains
        # eliminate chains of length 2
        chains = [chain for chain in chains if chain not in original_chains]
        return chains

    def restore_unary_rules(self, trees: List[Tree]):
        """
        add all possible unary rules back
        :param trees: trees using the original CFG
        :return: trees with unary rules restored
        """
        result_trees = []
        for tree in trees:
            if type(tree) == Tree:
                root = tree.label()
                children = [t for t in tree]
                children_roots = []
                for t in tree:
                    if type(t) == Tree:
                        children_roots.append(t.label())
                    else:
                        children_roots.append(t)

                # bottom-up restoration of unary rules by recursion
                # all possibilities of children found by cartesian product of possibilities for each rule
                children_restored = [self.restore_unary_rules([child]) for child in children]
                children_possibilities = list(product(*children_restored))
                for children in children_possibilities:
                    top_rule = Production(root, children_roots)
                    # directly add the tree, if restoration is not necessary
                    if top_rule in self.grammar.productions():
                        result_trees.append(Tree(root, children))
                    # consider all possibilities of restoration for the top rule
                    for chain in self.unary_chains:
                        if chain[0] == root and list(chain[-1]) == list(children_roots):
                            # insert chain between root and children
                            new_tree = Tree(chain[-2], children)
                            for i in range(-3, -len(chain)-1, -1):
                                new_tree = Tree(chain[i], [new_tree])
                            result_trees.append(new_tree)
            else:
                result_trees.append(tree)
        return result_trees

    def remove_added_nonterminals(self, tree: Tree):
        """
        Remove the added nonterminals in a tree
        e.g.
        ( A (C 'a') (1 (C 'a') (0 'c')) ) will be
        ( A (C 'a') (C 'a') 'c' ),

        :param tree: a tree represented by CNF of the CFG
        :return: the equivalent tree with added nonterminals eliminated
        """
        if type(tree) != Tree:
            return tree

        # recursively remove added nonterminals from bottom to top layers
        new_children = []
        for child in tree:
            child = self.remove_added_nonterminals(child)
            if type(child) == Tree and child.label() in self.added_nonterminals:
                for grandchild in child:
                    new_children.append(grandchild)
            elif type(child) == Tree:
                new_children.append(child)
            else:
                new_children.append(child)
        return Tree(tree.label(), new_children)

    def generate_cnf(self):
        """
        convert the input CFG into Chomsky Normal Form
        :return: None
        """
        self.grammar_cnf = self.grammar
        # step 1 generate new start symbol.
        # step 1 ignored because in cfg for natural languages
        # the start symbol will not be on the rhs of any rule

        # step 2 remove null rules and unary rules
        # I do not consider null rules to be valid in my grammar
        self.grammar_cnf = self.remove_null_rules(self.grammar_cnf)
        self.grammar_cnf = self.remove_unary_rules(self.grammar_cnf)

        # step 3
        self.grammar_cnf = self.separate_terminals(self.grammar_cnf)
        # step 4
        self.grammar_cnf = self.separate_nonterminals(self.grammar_cnf)
        return

    def remove_null_rules(self, grammar: CFG) -> CFG:
        """
        remove null rules such as
        A ->
        :param grammar:
        :return:
        """
        start = grammar.start()
        new_rules = [rule for rule in grammar.productions() if len(rule.rhs()) > 0]
        return CFG(start, new_rules)

    def remove_unary_rules(self, grammar: CFG) -> CFG:
        """
        eliminate unary rules
        e.g.
        A -> B, B -> C, C -> ? will become A -> ?

        after the transformation.
        :param grammar: a CFG
        :return: the equivalent CFG with terminals separated
        """
        start = grammar.start()
        need_to_remove = True
        removed_unary_rules = set()
        second_part = set()
        while need_to_remove:
            new_rules = set()
            removed_unary_rules_before = set(removed_unary_rules)
            for rule in grammar.productions():
                rhs = rule.rhs()
                if rule.is_nonlexical() and len(rhs) == 1:
                    # remove A -> B
                    removed_unary_rules.add(rule)
                    redundant_symbol = rhs[0]
                    lhs = rule.lhs()
                    for rule_to_change in grammar.productions(lhs=redundant_symbol):
                        # record B -> ?
                        second_part.add(rule_to_change)
                        # add the glued rule A -> ?
                        new_rules.add(Production(lhs, rule_to_change.rhs()))
            # add glued rules
            grammar = CFG(start, list(new_rules.union(grammar.productions())))
            # no change means all removals have been done, or no need to remove at all
            if removed_unary_rules_before == removed_unary_rules:
                need_to_remove = False
        # now filter out the rules that are redundant
        # the lhs of rules in second_part might appear on the rhs of some CNF rules,
        # so not necessary to remove second_part
        new_rules = [rule for rule in grammar.productions() if rule not in removed_unary_rules]
        new_grammar = CFG(start, new_rules)
        original_rules = set(self.grammar.productions())
        self.removed_unary_rules = removed_unary_rules.intersection(original_rules).union(second_part.intersection(original_rules))
        return new_grammar

    def separate_terminals(self, grammar: CFG) -> CFG:
        """
        make terminals generated by unitary rules
        e.g.
        A -> aBc where a,c are terminals will be
        A -> 0B1
        0 -> a
        1 -> c
        after the transformation, where 0 and 1 are newly added nonterminals
        :param grammar: a CFG
        :return: the equivalent CFG with terminals separated
        """
        start = grammar.start()
        new_rules = []
        for rule in grammar.productions():
            rhs = rule.rhs()
            if rule.is_lexical() and len(rhs) > 1:
                replaced_rhs = list(rhs)
                for i, symbol in enumerate(rhs):
                    if type(symbol) != Nonterminal:
                        new_nonterminal = self.create_nonterminal()
                        replaced_rhs[i] = new_nonterminal
                        new_rules.append(Production(new_nonterminal, [symbol]))
                new_rules.append(Production(rule.lhs(), replaced_rhs))
            else:
                new_rules.append(rule)

        return CFG(start, new_rules)

    def separate_nonterminals(self, grammar: CFG):
        """
        make each rule generate at most 2 nonterminals
        e.g.
        A -> BCDEF will become
        A -> B0
        0 -> C1
        1 -> D2
        2 -> EF
        after the transformation, where 0 and 1 are newly added nonterminals
        :param grammar: a CFG
        :return: the equivalent CFG with terminals separated
        """
        start = grammar.start()
        new_rules = []
        for rule in grammar.productions():
            rhs = rule.rhs()
            if rule.is_nonlexical() and len(rhs) > 2:
                new_nonterminal = self.create_nonterminal()
                # rhs: the first symbol and the rest
                new_rules.append(Production(rule.lhs(), [rhs[0], new_nonterminal]))
                for i in range(1,len(rhs)-2):
                    last_added_nonterminal = new_nonterminal
                    new_nonterminal = self.create_nonterminal()
                    # rhs: the i-th symbol and the rest
                    new_rules.append(Production(last_added_nonterminal, [rhs[i], new_nonterminal]))
                # rhs: the last two symbols
                new_rules.append(Production(new_nonterminal, [rhs[-2], rhs[-1]]))
            else:
                new_rules.append(rule)

        return CFG(start, new_rules)

    def create_nonterminal(self) -> Nonterminal:
        """
        create a nonterminal symbol, and increment the counter for number of created nonterminals
        :return: the created new nonterminal
        """
        new_nonterminal = Nonterminal(self.nonterminal_counter)
        self.nonterminal_counter += 1
        self.added_nonterminals.append(new_nonterminal)
        return new_nonterminal


def parse_sentences(parser: CykParser, sentences: List[str]):
    all = []
    all_true = True
    for s in sentences:
        trees = parser.parse(s)
        all.append(trees)
        print(f"Sentence: '{s}'")
        for t in trees:
            t.pretty_print(unicodelines=True, nodedist=2)
            print(str(t)+'\n')
        if len(trees) == 0:
            print("The sentence is not accepted.\n")
            all_true = False
    if all_true:
        print("All accepted.")
    return all


if __name__=='__main__':
    with open('french-grammar.txt') as f:
        lines = f.readlines()
    grammar = CFG.fromstring(lines)
    parser = CykParser(grammar)
    s_accept = [
        'je regarde la television',
        'tu regardes la television',
        'il regarde la television',
        'nous regardons la television',
        'vous regardez la television',
        'ils regardent la television',
        'tu ne regardes pas la television',
        'il la regarde',
        'Jonathan aime le petit chat',
        'Jonathan aime les chats noirs',
        'je aime le Canada',
        'le beau chat le mange',
        'les aides aiment Montreal',
         ]

    s_reject = [
        'je mangent le poisson',
        'les noirs chats mangent le poisson',
        'la poisson mangent les chats',
        'je mange les',
        'le chat'
    ]
    print("The following sentences should be accepted")
    parse_sentences(parser, s_accept)
    print("\nThe following sentences should be rejected")
    parse_sentences(parser, s_reject)