(define (problem blocksworld3)
    (:domain blocksworld_without_robot)
    (:objects
        yellow_block - block
        green_block - block
        pink_block - block
        purple_block - block
        red_block - block

    )
    (:init
        (ontable yellow_block)
        (ontable green_block)
        (ontable pink_block)
        (ontable red_block)
        (clear yellow_block)
        (clear green_block)
        (clear pink_block)
        (clear purple_block)
        (on purple_block red_block)
        (handempty)
    )
    (:goal (and (on yellow_block green_block) (on green_block pink_block) (on red_block purple_block) (clear red_block) (clear yellow_block)))
)