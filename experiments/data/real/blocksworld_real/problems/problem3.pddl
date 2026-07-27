(define (problem blocksworld_real3)
  (:domain blocksworld_real)
  (:objects
    red - block
    grey - block
    blue - block
    yellow - block
  )
  (:init
    (clear blue)
    (clear yellow)
    (handempty)
    (on grey red)
    (on blue grey)
    (ontable red)
    (ontable yellow)
  )
  (:goal (and
    (on yellow red)
    (on red grey)
    (on grey blue)
  ))
)
