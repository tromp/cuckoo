my $maxcycles = 0;
my $nonce = 0;
my $maxnonce = 0;
my $nnonces = 0;
my $maxcycles = 0;
my $ncycles = 0;
my @count;
while (<>) {
  if (/^nonce (\d+)/) {
    $nonce = $1;
    $nnonces += 1;
  } elsif (/^Time/ || /^findcycles/) {
    if ($ncycles > $maxcycles) {
      $maxnonce = $nonce;
      $maxcycles = $ncycles;
    }
    $ncycles = 0;
  } elsif (/(\d+)-cycle found/) {
    $ncycles += 1;
    $count[$1/2]++;
  }
}
for $i (1..$#count) {
  my $c = $count[$i];
  my $i2 = 2 * $i;
  my $f = $c * $i2 / $nnonces;
  print "$i2 $c $f\n" if $c;
}
printf "$nnonces nonces $maxcycles cycles at nonce $maxnonce\n";
