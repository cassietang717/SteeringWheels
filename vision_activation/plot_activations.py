from utils import plot_pca_comparison, load_chunks

gt_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest_gt_layer_wise_*.npy"
gt_layer_wise_activations = load_chunks(gt_layer_wise_pattern)
hallucinated_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest_hallucinated_layer_wise_*.npy"
hallucinated_layer_wise_activations = load_chunks(hallucinated_layer_wise_pattern)

gt_head_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest_gt_head_wise_*.npy"
gt_head_wise_activations = load_chunks(gt_head_wise_pattern)
hallucinated_head_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest_hallucinated_head_wise_*.npy"
hallucinated_head_wise_activations = load_chunks(hallucinated_head_wise_pattern)

plot_pca_comparison(gt_layer_wise_activations, hallucinated_layer_wise_activations, "HaloQuest_layer_wise", layer_num=33)
plot_pca_comparison(gt_head_wise_activations, hallucinated_head_wise_activations, "HaloQuest_head_wise", layer_num=32)