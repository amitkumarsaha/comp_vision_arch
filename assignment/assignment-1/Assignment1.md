# Aim:
- Explain the modern design pattern in computer vision models: 
  - Pretrained backbone + task-specific head
- How it is used in both supervised and self-supervised (SSL) settings. 
- Where this pattern works well and where it is less suitable. 

# Tasks
Write a short report that addresses the following points. You may use Meta Segment Anything Model (SAM) and Distillation with No Labels (DINO) as examples, but you can also mention other models (CLIP, Etc. ).
1. Backbone + head: basic concepts
   - Explain, in your own words:
     - What is meant by a backbone in modern vision models?
     -	What is meant by a head?
   - Give at least two concrete examples from the literature where a backbone is reused with different heads (e.g. classification, detection, segmentation, depth).
   - Include one simple diagram that illustrates “backbone + head” for any model of your choice.
2. Supervised vs self-supervised pretraining (be concrete)
   - Explain the difference between:
       1.	Supervised pretraining, and
       2.	Self-supervised pretraining (SSL)
       For this section, you must describe 1–2 concrete strategies for each.
       - Supervised pretraining
         1. Briefly describe at least one specific supervised pretraining strategy, for example:
            - Training a backbone for image classification on ImageNet (e.g. VGG, ResNet).
            - Or training a backbone jointly with a head for object detection on COCO (e.g. Faster/Mask R-CNN).
         2. For each supervised strategy you mention, answer:
           - What is the training signal (what labels are used)?
           - What kind of features are encouraged (e.g. category-level, object-level, etc.)?
       - Self-supervised pretraining (SSL)
           1. Briefly describe 1–2 specific SSL families, for example:
              - Siamese / contrastive approaches
                - Such as SimCLR, MoCo, BYOL, SimSiam, etc.
                -	Explain the idea of using two “views” of the same image and encouraging similar representations.
              - DINO-style self-distillation
                -	Explain at a high level how a “student” network learns from a “teacher” without labels.
           2. For the SSL strategies you mention, answer:
              -	What is the training signal (what is being matched or contrasted, if there are no labels)?
              -	What kind of features are encouraged (e.g. invariance to augmentations, grouping similar images, etc.)?
           3. Comparison
              - Explain one key difference between supervised and SSL pretraining in terms of:
                - The source of supervision, and
                - The type of representations learned.
              - Discuss one advantage and one limitation of using SSL backbones for downstream tasks (compared to purely supervised ones).
3. Case study: SAM (Segment Anything Model) as backbone + head
   - Using SAM as a case study (at a high level):
     - Identify:
       - Which part acts as the image backbone.
       - Which part acts as the task-specific head (or heads).
       - What role prompts (points / boxes / masks) play in its design.
     - Explain how SAM illustrates the idea:
       - “Train a powerful backbone once, then reuse it for many segmentation-like tasks by changing the prompts and heads.”
       Implementation details are not required; focus on a clear conceptual description.
4. Where generic backbones are less suitable
   - Give at least one example of a computer vision task where using a generic “ImageNet-style” backbone as a frozen feature extractor is not straightforward or not clearly ideal.
     1. Examples (you may choose one of these or another reasonable task):
        - Image super-resolution
        - Image denoising / deblurring
        - Raw sensor denoising / demosaicing
        - Optical flow estimation
   - For your chosen task, try to answer:
     - Why might high-level, invariant features from models like ResNet, DINO, or SAM not be enough for this task?
     - What kind of information or inductive bias is needed instead (for example, very fine pixel-level structure, knowledge of the degradation process, or physics of the imaging system)?
# Format and length
- Length: about 3–4 pages (excluding references).
- Style: clear, concise prose; diagrams encouraged (must have: one simple backbone+head diagram).
- References: cite the main models or methods you mention (e.g. ResNet, SimCLR/MoCo/BYOL/SimSiam, DINO, SAM); any standard citation format is acceptable.
# Assessment
- Conceptual understanding of backbone + head and examples: 30%
- Explanation of supervised vs self-supervised pretraining, with 1–2 concrete strategies each: 30%
- SAM case study and mapping to backbone/head/prompt view: 20%
- Discussion of limitations / low-level tasks: 20%