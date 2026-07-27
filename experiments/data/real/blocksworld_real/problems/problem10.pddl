(define (problem blocksworld_real10)
  (:domain blocksworld_real)
  (:objects
    black - block
    blue - block
    yellow - block
    red - block
  )
  (:init
    (clear black)
    (clear blue)
    (clear yellow)
    (holding red)
    (handfull)
    (ontable black)
    (ontable blue)
    (ontable yellow)
  )
  (:goal (and
    (on red yellow)
    (on yellow blue)
  ))
)
