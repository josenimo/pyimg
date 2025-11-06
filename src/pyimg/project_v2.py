import dask.array as da
import numpy as np
from typing import List, Dict, Set

def _validate_map(
    channel_map: Dict[int, List[int]], 
    num_in_channels: int
) -> int:
    """Helper to validate the channel map."""
    if not channel_map:
        raise ValueError("channel_map dictionary cannot be empty.")

    num_out_channels = len(channel_map)
    if num_out_channels > 3:
        print(f"Warning: Output has {num_out_channels} channels, which is > 3.")

    # Check that output keys are contiguous 0-indexed (0, 1, 2...)
    expected_out_keys = set(range(num_out_channels))
    if set(channel_map.keys()) != expected_out_keys:
        raise ValueError(
            f"channel_map keys must be a contiguous range from 0 to N-1. "
            f"Got: {sorted(list(channel_map.keys()))}"
        )

    # Check for exclusivity and index bounds
    used_in_indices: Set[int] = set()
    all_in_indices: List[int] = []

    for out_idx in range(num_out_channels):
        in_indices = channel_map[out_idx]
        if not in_indices:
            raise ValueError(f"Input channel list for output channel {out_idx} is empty.")
        
        all_in_indices.extend(in_indices)
        for idx in in_indices:
            if not (0 <= idx < num_in_channels):
                raise IndexError(
                    f"Input channel index {idx} (for output channel {out_idx}) "
                    f"is out of bounds. Input image has {num_in_channels} channels (0 to {num_in_channels-1})."
                )
            if idx in used_in_indices:
                raise ValueError(
                    f"Input channel {idx} is used more than once (violates exclusivity). "
                    f"It's requested for output channel {out_idx} and a previous one."
                )
            used_in_indices.add(idx)
            
    return num_out_channels


def merge_channels_dask(
    img_stack: da.Array,
    channel_map: Dict[int, List[int]],
    proj: str = "sum",
) -> da.Array:
    """
    Lazily merge subsets of channels from a (C, Y, X) Dask array.

    Parameters:
    - img_stack (da.Array): The input lazy Dask array with shape (C, Y, X).
    - channel_map (Dict[int, List[int]]):
        A map where keys are output channel indices (0, 1, 2...) and
        values are lists of 0-based *input* channel indices to merge.
        Example: {0: [1, 2, 5], 1: [0, 4]}
    - proj (str, optional): The projection method. Default is "sum".
        - "sum": Sum the pixel values of the selected channels.
        - "mean": Average the pixel values.
        - "max": Take the max pixel value (Max Intensity Projection).

    Returns:
    - output_stack (da.Array):
        A lazy Dask array with shape (N, Y, X), where N is len(channel_map).
    """

    # --- 1. Validation ---
    if img_stack.ndim != 3:
        raise ValueError(f"Input array must be 3D (C, Y, X). Got {img_stack.ndim} dimensions.")
    
    num_in_channels = img_stack.shape[0]
    num_out_channels = _validate_map(channel_map, num_in_channels)
    
    # --- 2. Build Lazy Graph ---
    output_channels_list = []
    
    # Iterate from 0 to N-1 to ensure correct output order
    for out_idx in range(num_out_channels):
        in_indices = channel_map[out_idx]
        
        # 1. Lazily select the (C_sub, Y, X) stack for this group
        # This is Dask's "fancy indexing"
        selected_channels = img_stack[in_indices, :, :]
        
        # 2. Lazily project along the channel axis (axis=0)
        if proj == "sum":
            # Project (C_sub, Y, X) -> (Y, X)
            merged_channel = selected_channels.sum(axis=0)
            # The original function clipped sums at 1, implying 0-1 float data.
            # We will do the same. Use .clip() as Dask arrays are immutable.
            # If you are using uint16, you might want to clip to 65535
            # or (better) change the dtype to float32 *before* summing.
            
            # For this example, we assume float and clip to 1.0.
            # If your data is integer, you should reconsider this clipping.
            # merged_channel = merged_channel.clip(max=1.0)
            
        elif proj == "mean":
            merged_channel = selected_channels.mean(axis=0)
            
        elif proj == "max":
            merged_channel = selected_channels.max(axis=0)
            
        else:
            raise ValueError(f"Unknown projection method: {proj}. Use 'sum', 'mean', or 'max'.")
        
        output_channels_list.append(merged_channel)
        
    # --- 3. Lazily stack all (Y, X) results into a final (N, Y, X) array ---
    output_stack = da.stack(output_channels_list, axis=0)
    
    return output_stack

# --- Example Usage ---
if __name__ == "__main__":
    
    # 1. Simulate a large (10, 500, 500) OME-TIF
    # We use chunks to make it lazy and process it in blocks.
    # Chunks are (channels, y_pixels, x_pixels)
    in_shape = (10, 500, 500)
    chunks = (2, 250, 250) 
    
    print(f"--- Simulating lazy input array ---")
    print(f"Input Shape: {in_shape}")
    print(f"Chunking: {chunks}\n")
    
    # Create a lazy random Dask array (pretends to be our TIF)
    lazy_input_stack = da.random.random(in_shape, chunks=chunks)
    
    print("Lazy input array representation:")
    print(lazy_input_stack)
    print("\n" + "="*30 + "\n")
    
    # 2. Define the channel map (using 0-based indexing)
    # As per your example:
    # - Output 0: Merge input channels 1, 2, 3, 5, 9
    # - Output 1: Merge input channels 4, 6, 7
    # - Output 2: Merge input channel 0 (e.g., DAPI)
    channel_map = {
        0: [1, 2, 3, 5, 9],  # Output channel 0 is sum of these
        1: [4, 6, 7],       # Output channel 1 is sum of these
        2: [0]              # Output channel 2 is just channel 0
    }
    
    print(f"--- Building lazy merge graph ---")
    print(f"Channel Map: {channel_map}")
    print(f"Projection: 'sum'\n")

    # 3. Build the lazy Dask graph
    # This step is instant; no computation is done.
    lazy_output = merge_channels_dask(
        lazy_input_stack, 
        channel_map, 
        proj="sum"
    )
    
    print("Lazy output array representation (no computation yet):")
    print(lazy_output)
    print(f"Expected output shape: ({len(channel_map)}, {in_shape[1]}, {in_shape[2]})")
    print("\n" + "="*30 + "\n")

    # 4. Trigger computation
    # This is when Dask reads the chunks and performs the parallel sum.
    print("--- Triggering computation... ---")
    final_merged_image = lazy_output.compute()
    print("--- Computation complete! ---")
    
    print(f"\nFinal computed image shape: {final_merged_image.shape}")
    print(f"Final computed image type: {type(final_merged_image)}")

    # --- Example of an error (violating exclusivity) ---
    try:
        print("\n" + "="*30 + "\n")
        print("--- Testing validation (violating exclusivity) ---")
        bad_map = {
            0: [1, 2, 3],
            1: [3, 4, 5]  # Error: Channel 3 is used twice
        }
        merge_channels_dask(lazy_input_stack, bad_map)
    except ValueError as e:
        print(f"Caught expected error: {e}")