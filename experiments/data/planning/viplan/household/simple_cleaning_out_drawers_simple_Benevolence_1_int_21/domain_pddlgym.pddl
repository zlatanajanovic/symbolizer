(define (domain igibson)
  (:requirements :strips :typing :negative-preconditions)

  (:types
    container movable - object
    sliceable slicer - movable
  )

  (:predicates
    (reachable ?o - object)
    (holding ?m - movable)
    (open ?c - container)
    (ontop ?o1 - object ?o2 - object)
    (inside ?o - object ?c - container)
    (nextto ?o1 - object ?o2 - object)
    (sliced ?s - sliceable)
  )

  (:action grasp
    :parameters (?m - movable)
    :precondition (and (reachable ?m))
    :effect (and (holding ?m))
  )

  (:action place-on
    :parameters (?m - movable ?o2 - object)
    :precondition (and (holding ?m) (reachable ?o2))
    :effect (and (ontop ?m ?o2) (not (holding ?m)))
  )

  (:action place-next-to
    :parameters (?m - movable ?o2 - object)
    :precondition (and (holding ?m) (reachable ?o2))
    :effect (and (nextto ?m ?o2) (not (holding ?m)))
  )

  (:action place-inside
    :parameters (?m - movable ?c - container)
    :precondition (and (holding ?m) (reachable ?c) (open ?c))
    :effect (and (inside ?m ?c) (not (holding ?m)))
  )

  (:action open-container
    :parameters (?c - container)
    :precondition (and (reachable ?c) (not (open ?c)))
    :effect (and (open ?c))
  )

  (:action close-container
    :parameters (?c - container)
    :precondition (and (reachable ?c) (open ?c))
    :effect (and (not (open ?c)))
  )

  (:action navigate-to
    :parameters (?o - object)
    :precondition (and (not (reachable ?o)))
    :effect (and (reachable ?o))
  )

  (:action slice
    :parameters (?o - sliceable ?s - slicer)
    :precondition (and (holding ?s) (reachable ?o) (not (sliced ?o)))
    :effect (and (sliced ?o))
  )
)
