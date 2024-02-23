import anytree
from anytree import RenderTree, findall_by_attr
from transformers import AutoTokenizer

class EntityTree():
    def __init__(self, tree : anytree.node.node.Node):
        self.tree = tree

    def show(self) -> None:
        for pre, fill, node in RenderTree(self.tree):
            print("%s%s" % (pre, node.name))

    def retrieve(self, entities_in_query :list[str]) -> dict:
        ls = []
        for entity in entities_in_query:
            ent = findall_by_attr(self.tree, entity)[0]
            #print(anytree.search.findall(self.tree, filter_=lambda node: ent in node.path))
            entity_info = f"{entity} belongs to {ent.parent.name} and contains all following entities: {[node.name for node in ent.children]}."
            ls.append(entity_info)
        return EntityTree.pre_process(ls)

    @classmethod
    def pre_process(self, list : list[str]) -> dict:
        # convert list  of 3 strings to dict of 2 key (input_id, attention_mask) and 3 values for each key
        entities_encoded = {}
        tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-base")
        encoded = tokenizer(list, return_tensors="pt", padding=True)
        entities_encoded["input_ids"] = encoded["input_ids"]
        entities_encoded["attention_mask"] = encoded["attention_mask"]
        entities_encoded["entity_info"] = list

        return entities_encoded
