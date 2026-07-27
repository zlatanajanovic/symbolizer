(define (problem blocksworld_real9)
  (:domain blocksworld_real)
  (:objects
    red - block
    black - block
    yellow - block
  )
  (:init
    (clear red)
    (clear black)
    (clear yellow)
    (handempty)
    (ontable red)
    (ontable black)
    (ontable yellow)
  )
  (:goal (and
    (on yellow black)
    (on black red)
  ))
)
