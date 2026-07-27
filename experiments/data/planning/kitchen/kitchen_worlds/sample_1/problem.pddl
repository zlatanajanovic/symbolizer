(define (problem pick_braiserlid)
    (:domain kitchen_worlds)
    (:objects
        robot - robot
        braiserlid medicine sweetpotato zucchini - item
        basin_bottom braiser_bottom counter_left counter_right fridge_shelf - surface
        braiser_area counter_left_area counter_right_area fridge_area sink_area - location
    )
    (:init
        (at-robot robot braiser_area)
        (handempty robot)
        (graspable braiserlid)
        (graspable medicine)
        (graspable sweetpotato)
        (graspable zucchini)
        (edible sweetpotato)
        (edible zucchini)
        (on braiserlid braiser_bottom)
        (on medicine counter_right)
        (on sweetpotato counter_left)
        (on zucchini basin_bottom)
        (at-surface basin_bottom sink_area)
        (at-surface braiser_bottom braiser_area)
        (at-surface counter_left counter_left_area)
        (at-surface counter_right counter_right_area)
        (at-surface fridge_shelf fridge_area)
        (heating-surface braiser_bottom)
        (cleaning-surface basin_bottom)
    )
    (:goal (and
        (holding robot braiserlid)
    ))
)
