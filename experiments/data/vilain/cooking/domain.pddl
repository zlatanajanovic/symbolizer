; Kitchen locations: cutting board, bowl, counter (ingredients default location)
; Objects: robot(s), ingredients (tomato, cucumber, dressing bottle), containers (bowl, cutting board)?, tools (knife, spatula, spoon)
; conditions/predicates states (whole/sliced/slice-type*, clean/dirty, empty/full)?
; Actions: pick (with tool? from to ), place( from to ), slice (from state to state), pour

;Header and description

(define (domain cooking)

    ;remove requirements that are not needed
    (:requirements :typing :adl)

    (:types ;todo: enumerate types and their hierarchy here, e.g. car truck bus - vehicle
        location ; objects can be in a location
        robot ; we have to robot arms, so we can define each robot's action and states
        vegetable tool - thing ; objects that can be manipulated (pick/place/...)
        )

    ; un-comment following line if constants are needed
    ; (:constants )

    (:predicates ;todo: define predicates here
        ; General thing predicates 
        (available ?obj - thing) ; True is thing is pickable
        (is-whole ?veg - vegetable) ; True if vegetable is intact
        (is-sliced ?veg - vegetable) ; True if vegetable is sliced

        ; Robot predicates
        (free ?robot - robot)
        (carry ?robot - robot ?obj - thing)
        (can-cut ?tool - tool)

        ; Location predicates
        (at ?obj - thing ?loc - location) ; Describes the location of the thing
        (is-workspace ?loc - location) ; can be used for slicing

    )

    ; (:functions 
    ; )

    ;define actions here
    (:action pick
        :parameters (?robot - robot ?obj - thing ?pick_loc - location)
        :precondition(
      and
            (available ?obj)
            (free ?robot)
            (at ?obj ?pick_loc)
        )
        :effect(
      and
            (not (available ?obj)) ; thing
            (carry ?robot ?obj) ; robot
            (not (free ?robot)) ; robot
            (not (at ?obj ?pick_loc)) ; location
        )
    )

    (:action place
        :parameters (?robot - robot ?obj - thing ?place_loc - location)
        :precondition(
      and
            ; robot - thing
            (carry ?robot ?obj)
            ; robot
            (not (free ?robot))
            ; location - thing
            (not (at ?obj ?place_loc))
        )
        :effect(
      and
            (available ?obj) ; thing
            (free ?robot) ; robot
            (at ?obj ?place_loc) ; location
            (not (carry ?robot ?obj)) ; robot - thing
        )
    )

    (:action slice
        :parameters (?robot - robot ?veg - vegetable ?tool - tool ?loc - location)
        :precondition(
      and
            ; robot - tool
            (carry ?robot ?tool)
            ; tool
            (can-cut ?tool)
            ; vegetable
            (is-whole ?veg)
            (not (is-sliced ?veg))
            ; location
            (is-workspace ?loc)
            ; location - thing
            (at ?veg ?loc)
        )
        :effect(
      and
            ; vegetable
            (is-sliced ?veg)
            (not (is-whole ?veg))
        )
    )
)
