#!/usr/bin/perl
use strict;

sub ent1 {
  my ($p) = @_;
  return -$p*log($p)/log(2.0);
}

sub ent {
  my ($p) = @_;
  return ent1($p) + ent1(1-$p);
}

my $f = 0;
my @comp;
while (<>) {
  next unless /8 part V0 load (\d+)/;
  my $load = $1;
  my $nc = 3200 * ++$f/100.0;
  my $nunc = 3200-$nc;
  my $p = ($load-$nunc)/$nc;
  my $e = ent($p);
  $comp[$f] = $e;
  # printf("%2d %4d %3.4lf %3.4lf\n",$f,$load,$p,$e);
}
for my $mp (19..99) { # memory percentage
  printf("\nmp = %2d ",$mp);
  my $i;
  my $np = $mp;   # nonce percentage
  for ($i=0; $np < 100;  $i++) {
    my $c = $comp[$mp];
    $np += (1-$c)*$mp;
    $np = int($np);
    printf("%2d ",$np);
  }
}
