(define (problem blocksworld7)
    (:domain blocksworld_without_robot)
    (:objects
        yellow_block - block
        pink_block - block
        green_block - block
        red_block - block
        purple_block - block
        blue_block - block

    )
    (:init
        (ontable green_block)
        (ontable purple_block)
        (ontable blue_block)
        (clear yellow_block)
        (clear red_block)
        (clear blue_block)
        (on yellow_block pink_block)
        (on pink_block green_block)
        (on red_block purple_block)
        (handempty)
    )
    (:goal (and (on yellow_block pink_block) (on pink_block green_block) (on green_block red_block) (on red_block purple_block) (on purple_block blue_block)))
)