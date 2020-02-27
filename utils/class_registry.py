from .trie import Trie


_REGISTRY = Trie()

def register_cls(identifier):
    def add_class(cls):
        _REGISTRY.add(identifier, cls)
        return cls
        
    return add_class

def find_cls(identifier, default_value=None):
    return _REGISTRY.get(identifier, default_value)
