

from chunk_methods.chunks import  ChunkHierarchicalParagraph, ChunkHierarchicalSection, ChunkMarkdown, ChunkSemantic, ChunkSlidingWindow


teste = ChunkHierarchicalParagraph(
    dataset_path="/workspaces/tcc/10lines.csv"
)

print(teste.execute())