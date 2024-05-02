def remove_prefix_from_state_dict(original_state_dict, prefix_to_remove = 'resnet18.'):
    new_state_dict = {}
    # Iterate through the keys in the original state dictionary
    for key, value in original_state_dict.items():
        # Check if the key starts with the specified prefix
        if key.startswith(prefix_to_remove):
            # Create a new key without the prefix
            new_key = key[len(prefix_to_remove):]
            # Add the entry to the new state dictionary
            new_state_dict[new_key] = value
        else:
            # If the key doesn't start with the prefix, keep it unchanged
            new_state_dict[key] = value
    return new_state_dict
