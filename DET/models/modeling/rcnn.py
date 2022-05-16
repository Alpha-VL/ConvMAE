from .postprocessing import detector_postprocess


def _postprocess(instances, batched_inputs, image_sizes):
    """Rescale the output instances to the target size."""
    # note: private function; subject to changes
    processed_results = []
    for results_per_image, input_per_image, image_size in zip(
        instances, batched_inputs, image_sizes
    ):
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])
        r = detector_postprocess(results_per_image, height, width)
        processed_results.append({"instances": r})
    return processed_results
