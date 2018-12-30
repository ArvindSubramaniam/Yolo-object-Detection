def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    """

    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs
    

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1, keepdims=False)
    

    # Step 3: Creating a filtering mask based on "box_class_scores" by using "threshold". The mask will be true for boxes that are to be retained
    
    filtering_mask = box_class_scores >= threshold
    

    # Step 4: Apply the mask to scores, boxes and classes
    
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    #

    return scores, boxes, classes
