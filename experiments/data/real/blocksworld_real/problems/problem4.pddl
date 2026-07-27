(define (problem blocksworld_real4)
  (:domain blocksworld_real)
  (:objects
    red - block
    grey - block
    yellow - block
    blue - block
  )
  (:init
    (clear grey)
    (clear yellow)
    (holding blue)
    (handfull)
    (on grey red)
    (ontable red)
    (ontable yellow)
  )
  (:goal (and
    (on yellow grey)
  ))
)
