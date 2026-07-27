(define (problem cook_zucchini_rollout_5)
    (:domain kitchen_worlds)
    (:objects
        robot - robot
        braiserlid medicine sweetpotato zucchini - item
        basin_bottom braiser_bottom counter_left counter_right fridge_shelf - surface
        braiser_area counter_left_area counter_right_area fridge_area sink_area - location
    )
    (:init
        (at-robot robot sink_area)
        (at-surface basin_bottom sink_area)
        (at-surface braiser_bottom braiser_area)
        (at-surface counter_left counter_left_area)
        (at-surface counter_right counter_right_area)
        (at-surface fridge_shelf fridge_area)
        (cleaned zucchini)
        (cleaning-surface basin_bottom)
        (clear basin_bottom)
        (clear fridge_shelf)
        (edible sweetpotato)
        (edible zucchini)
        (graspable braiserlid)
        (graspable medicine)
        (graspable sweetpotato)
        (graspable zucchini)
        (heating-surface braiser_bottom)
        (holding robot zucchini)
        (on braiserlid counter_right)
        (on medicine counter_left)
        (on sweetpotato braiser_bottom)
    )
    (:goal (and
        (cooked zucchini)
    ))
)
