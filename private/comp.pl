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

my $f = 20;
while (<>) {
  next unless /8 part V0 load (\d+)/;
  my $load = $1;
  my $nc = 3200 * $f/100.0;
  my $nunc = 3200-$nc;
  my $p = ($load-$nunc)/$nc;
  my $e = ent($p);
  print "$f $load $p $e\n";
  $f++;
}
