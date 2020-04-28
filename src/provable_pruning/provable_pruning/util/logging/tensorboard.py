"""Some utility functions to log to a tensorboard."""


def log_scalar(
    writer, global_tag, tag, value, step, n_idx=None, r_idx=None, s_idx=None
):
    """Log a scalar variable to tensorboard."""
    # generate the full tag
    tag_list = [global_tag]
    if n_idx is not None:
        tag_list.append("n{}".format(n_idx))
    if r_idx is not None:
        tag_list.append("r{}".format(r_idx))
    if s_idx is not None:
        tag_list.append("s{}".format(s_idx))
    tag_list.append(tag)

    full_tag = "/".join(tag_list)

    writer.add_scalar(tag=full_tag, scalar_value=value, global_step=step)
    writer.flush()


def log_sens_histogram(writer, global_tag, tag, n_idx, step, sensitivity):
    """Log the sensitivity histogramm to tensorboard."""
    # reshape into (numFeatures, numWeightsPerFeature)
    sensitivity = sensitivity.view(sensitivity.shape[0], -1)

    # cut of large values
    values = sensitivity[sensitivity < 0.5 * sensitivity.max()]
    # sometimes that doesn't work so let's keep athe original one
    if not values.numel():
        values = sensitivity

    # write to histogram
    log_histogram(
        writer=writer,
        global_tag=global_tag,
        tag=tag,
        n_idx=n_idx,
        values=values.data.cpu().numpy(),
        step=step,
    )


def log_histogram(writer, global_tag, tag, n_idx, values, step):
    """Log a histogram of the tensor of values to tensorboard."""
    full_tag = "/".join([global_tag, str(n_idx), tag])

    # add to writer
    writer.add_histogram(
        tag=full_tag, values=values, global_step=step, bins="auto"
    )
    writer.flush()


def log_image(writer, tag, image, step):
    """Log one image."""
    writer.add_image(
        tag=tag, img_tensor=image, global_step=step, dataformats="HWC"
    )
    writer.flush()


def log_images(writer, tag, images, step):
    """Log a list of images."""
    for i, img in enumerate(images):
        tag_i = f"{tag}/{i}"
        log_image(writer, tag_i, img, step)


def log_text(writer, tag, value, step):
    """Log text."""
    writer.add_text(tag=tag, text_string=value, global_step=step)
    writer.flush()
