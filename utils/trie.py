from collections import defaultdict

class Trie():

    class Node():
        def __init__(self, value=None):
            self.value = value
            self.children = {}

    def __init__(self):
        self.root = self.Node()
        self.count = 0

    def add(self, key, value):
        path = key.split('.')
        node = self.root
        for token in path:
            if token not in node.children:
                node.children[token] = self.Node()
            node = node.children[token]
        node.value = value
        self.count += 1

    def get(self, key, default_value=None):
        path = key.split('.')
        node = self.root
        for token in path:
            if token not in node.children:
                return default_value
            node = node.children[token]
        self.count -= 1
        return node.value
