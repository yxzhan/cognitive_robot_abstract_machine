---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(semantic_annotations_overview)=
# Semantic Annotations

Semantic annotations attach meaning to the bodies and regions in a world: that a body *is* a
handle, or that a collection of bodies *is* a drawer. Because they are plain, typed Python
classes, you can reason over them with the Entity Query Language and persist them through the ORM.

This part of the guide collects the semantic-annotation tutorials:

- [](semantic_annotations) — what semantic annotations are and how to define your own.
- [](semantic_annotation_composition) — composing annotations from structural parts with the
  unified `add` method and `composition_field`.
- [](semantic_annotation_factories) — convenience factories that spawn an annotation together with
  a body and generated geometry.
