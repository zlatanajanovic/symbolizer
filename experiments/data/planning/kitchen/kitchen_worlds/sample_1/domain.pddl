(define (domain kitchen_worlds)
    (:requirements :strips :typing)
    (:types
        item surface robot location
    )
    (:predicates
        (on ?i - item ?s - surface)
        (holding ?r - robot ?i - item)
        (handempty ?r - robot)
        (at-robot ?r - robot ?l - location)
        (at-surface ?s - surface ?l - location)
        (clear ?s - surface)
        (graspable ?i - item)
        (cleaning-surface ?s - surface)
        (heating-surface ?s - surface)
        (cleaned ?i - item)
        (cooked ?i - item)
        (edible ?i - item)
    )

    (:action move
        :parameters (?r - robot ?from - location ?to - location)
        :precondition (and
            (at-robot ?r ?from)
        )
        :effect (and
            (at-robot ?r ?to)
            (not (at-robot ?r ?from))
        )
    )

    (:action pick
        :parameters (?r - robot ?i - item ?s - surface ?l - location)
        :precondition (and
            (on ?i ?s)
            (handempty ?r)
            (at-robot ?r ?l)
            (at-surface ?s ?l)
            (graspable ?i)
        )
        :effect (and
            (holding ?r ?i)
            (not (on ?i ?s))
            (not (handempty ?r))
            (clear ?s)
        )
    )

    (:action place
        :parameters (?r - robot ?i - item ?s - surface ?l - location)
        :precondition (and
            (holding ?r ?i)
            (at-robot ?r ?l)
            (at-surface ?s ?l)
        )
        :effect (and
            (on ?i ?s)
            (handempty ?r)
            (not (holding ?r ?i))
            (not (clear ?s))
        )
    )

    (:action clean
        :parameters (?r - robot ?i - item ?s - surface ?l - location)
        :precondition (and
            (on ?i ?s)
            (cleaning-surface ?s)
            (at-robot ?r ?l)
            (at-surface ?s ?l)
            (handempty ?r)
            (edible ?i)
        )
        :effect (and
            (cleaned ?i)
        )
    )

    (:action cook
        :parameters (?r - robot ?i - item ?s - surface ?l - location)
        :precondition (and
            (on ?i ?s)
            (heating-surface ?s)
            (at-robot ?r ?l)
            (at-surface ?s ?l)
            (handempty ?r)
            (cleaned ?i)
        )
        :effect (and
            (cooked ?i)
        )
    )
)
