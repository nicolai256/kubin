import os
import uuid
from datetime import datetime
from PIL import PngImagePlugin

def create_png_info(seed, prompt, negative_prior_prompt, negative_decoder_prompt, num_steps, batch_size, guidance_scale, h, w, sampler, prior_cf_scale, prior_steps):
    # Create a new PngInfo object
    png_info = PngImagePlugin.PngInfo()

    # Add custom metadata to the image
    png_info.add_text("prompt", str(prompt))
    png_info.add_text("negative_prior_prompt", str(negative_prior_prompt))
    png_info.add_text("negative_decoder_prompt", str(negative_decoder_prompt))
    png_info.add_text("seed", str(seed))
    png_info.add_text("num_steps", str(num_steps))
    png_info.add_text("batch_size", str(batch_size))
    png_info.add_text("guidance_scale", str(guidance_scale))
    png_info.add_text("h", str(h))
    png_info.add_text("w", str(w))
    png_info.add_text("sampler", str(sampler))
    png_info.add_text("prior_cf_scale", str(prior_cf_scale))
    png_info.add_text("prior_steps", str(prior_steps))

    return png_info

def save_output(output_dir, task_type, images, seed, prompt, negative_prior_prompt, negative_decoder_prompt, num_steps, batch_size, guidance_scale, h, w, sampler, prior_cf_scale, prior_steps):
  output = []
  for img in images:
    path = f'{output_dir}/{task_type}'
    if not os.path.exists(path): os.makedirs(path)

    words = prompt.strip().split()
    stripped_prompt = '_'.join(words[:5])

    current_datetime = datetime.now()
    format_string = "%Y%m%d%H%M%S"
    formatted_datetime = current_datetime.strftime(format_string)

    name_of_file = formatted_datetime + "_" + stripped_prompt

    # name = f'{path}/{seed}-{uuid.uuid4()}.png'
    filename = f'{path}/{name_of_file}.png'
    while os.path.exists(filename):
      unique_id = current_datetime.microsecond
      filename = f'{formatted_datetime}_{unique_id}.jpg'

    img.save(filename, 'PNG', pnginfo=create_png_info(seed, prompt, negative_prior_prompt, negative_decoder_prompt, num_steps, batch_size, guidance_scale, h, w, sampler, prior_cf_scale, prior_steps))
    # add some stuff to add PngInfo to images with pil PngInfo
    output.append(filename)

  return output

 


