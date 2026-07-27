(define (problem blocksworld_real5)
  (:domain blocksworld_real)
  (:objects
    red - block
    grey - block
    white - block
    yellow - block
    blue - block
	black - block
  )
  (:init
    (clear grey)
    (clear white)
    (clear yellow)
    (holding blue)
    (handfull)
    (on grey red)
    (on white black)
    (ontable red)
    (ontable black)
    (ontable yellow)
  )
  (:goal (and
    (on blue yellow)
  ))
)
