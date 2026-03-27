# yq command used often
#

# extract entries with keys containing "rep0"
yq -y '[to_entries[] | select(.key | contains("rep0"))] | from_entries' motif_concept_pairs.yaml > motif_concept_pairs.rep0.yaml
