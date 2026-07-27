(define (problem blocksworld_real2)
  (:domain blocksworld_real)
  (:objects
    yellow - block
    white - block
    grey - block
    red - block
  )
  (:init
    (clear yellow)
    (clear red)
    (handempty)
    (on white grey)
    (on yellow white)
    (ontable grey)
    (ontable red)
  )
  (:goal (and
    (on yellow white)
    (on white grey)
    (on grey red)
  ))
)
