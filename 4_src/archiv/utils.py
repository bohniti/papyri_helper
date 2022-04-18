def create_info_subset(info, all=False, file_names=None, n=None, rand=False, label=None):
    if all:
        return info
    elif file_names is not None:
        if label is None:
            subset_info = info[info['names'].isin(file_names)]
            return subset_info
        else:
            test = info[info['names'].isin(file_names)]
            subset_info = test[test['labels'] == label]
            return subset_info
    # Print first n images
    elif n is not None and not rand:
        if label is None:
            subset_info = info.head(n)
            return subset_info
        else:
            test = info.head(n)
            subset_info = test[test['labels'] == label]
            return subset_info
    # Print n random images
    elif n is not None and rand:
        if label is None:
            subset_info = info.sample(n)
            return subset_info
        else:
            test = info.sample(n)
            subset_info = test[test['labels'] == label]
            return subset_info
